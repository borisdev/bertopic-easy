from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal, Optional, Union

from bertopic import BERTopic
from loguru import logger
from pydantic import BaseModel, Field
from rich import print
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({"info": "dim cyan", "warning": "magenta", "danger": "bold red"})
console = Console(theme=custom_theme)


class LabeledDoc(BaseModel):
    pos: Optional[int] = None
    doc: str
    label: int
    prob: Optional[float] = None
    llm_label: Optional[str] = None

class PipelineConfig(BaseModel):
    min_topic_size: int = 4
    sink_path: Path = Path("bertopic_easy_output.json")
    start: int = 0
    size: Union[int, None] = None
    batch_size: int = 100
    max_tokens: int = 200
    max_retries: int = 2
    reasoning_effort: Optional[Literal["high", "medium", "low"]] = "low"

class Pipeline(BaseModel):
    """
    step 1: bertopic - dbscan cluster with openai embeddings
    step 2: LLM helper - name clusters with LLM -- do whole group naming to avoid near duplicates
    step 3: LLM helper - classify outliers with LLM and reassign to clusters accordingly
    """

    config: PipelineConfig
    source_sentences: list[str] = []
    clusters: dict[int, list[LabeledDoc]] = {}
    pruned_clusters: dict[int, list[LabeledDoc]] = {}
    named_clusters: dict[str, list[LabeledDoc]] = {}
    named_clusters_w_merged_outliers: dict[str, list[LabeledDoc]] = {}

    class Config:
        arbitrary_types_allowed = True


    @classmethod
    def load(cls, *, config: PipelineConfig) -> Pipeline:
        file_path = config.sink_path
        logger.info(f"Loading {cls.__name__} from {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
        console.print(f"Loaded {cls.__name__} from {file_path}", style="info")
        return cls(**data)

    def save(self) -> None:
        sink_path = self.config.sink_path
        with open(sink_path, "w") as f:
            json_dump = self.model_dump_json(indent=2)
            f.write(json_dump)
        console.print(f"Saved to {sink_path}", style="info")

    async def run(self, source_sentences: list[str]) -> None:
        count = len(source_sentences)
        unique_docs = list(set(source_sentences))
        if count != len(unique_docs):
            raise ValueError(f"Duplicate docs found in source_sentences")
        self.source_sentences = source_sentences
        self.step1_embedding_clusters()
        self.step2_name_as_whole_w_llm()
        await self.step3_classify_outliers_w_llm(
            start=self.config.start,
            size=self.config.size,
            batch_size=self.config.batch_size,
            llm_name="o3-mini",
            max_tokens=self.config.max_tokens,
            max_retries=self.config.max_retries,
            timeout=None,
            reasoning_effort=self.config.reasoning_effort,
        )

    def step1_embedding_clusters(self) -> None:
        docs = self.source_sentences
        if len(docs) < 7:
            raise ValueError(f"Too few docs: {len(docs)}")
        embedding_model = "text-embedding-3-large"
        embeddings = get_embeddings_with_cache(texts=docs, model=embedding_model)
        topic_model = BERTopic(min_topic_size=config.min_topic_size)
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        labeled_docs = [
            LabeledDoc(doc=doc, label=label, prob=prob)
            for doc, label, prob in zip(docs, topics, probs)
        ]
        # over write any previous clusters
        self.clusters = defaultdict(list)
        for labeled_doc in labeled_docs:
            self.clusters[int(labeled_doc.label)].append(labeled_doc)
        report = {
            "total_docs": len(docs),
            "total_clusters": len(self.clusters),
            "b4_reduced_docs_in_outliers": len(self.clusters[-1]),
        }
        ## skip hierarchical topics for now
        ## binary tree where each parent node has two children, left and right
        ## Nice EDA visualizations
        # hierarchical_topics = topic_model.hierarchical_topics(docs)
        # topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        # tree = topic_model.get_topic_tree(hierarchical_topics)
        # df = topic_model.get_topic_info()
        # print(clusters)
        # print(df)
        self.save()
        print(self.clusters)
        # TODO save report for each step
        print(report)

    def step2_name_as_whole_w_llm(
        self,
    ) -> None:
        self.named_clusters = {}
        if len(self.clusters) == 0:
            raise ValueError("Clusters is None....run step1 first")


        class UnamedGroup(BaseModel):
            group_number: int
            docs: list[str]

        class UnamedGroups(BaseModel):
            unnamed_groups: list[UnamedGroup]

        unamed_groups = []
        for group_number, members in self.clusters.items():
            if group_number == -1:
                continue
            unamed_group = UnamedGroup(group_number=group_number, docs=[member.doc for member in members])
            unamed_groups.append(unamed_group)
        unamed_groups_json = UnamedGroups(unnamed_groups=unamed_groups).model_dump_json(indent=2)


        prompt = f"""

        Instructions
        ========================

        Provide a very informative research headline for each of the following unnamed groups of personal diet intervention outcomes.

        Make sure each headlines is distinct from the others.

        Get to the point quickly. Don't use unnecessary words. Don't use filler words. Don't use vague words.
        Don't use words that are not necessary to convey the meaning of the group such as overly general words
        or generic terms.

        People should be able to quickly understand what the research on personal diet intervention outcomes is about by reading the headline.

        Specifics in the name are good so people can quickly understand what the group is about.

        Make sure the name represents the common specifics of what makes its group different that the other groups.

        Your output should be a list of names, one for each cluster, in the same order as the clusters are presented in the input.

        In your output, each group headline must be in the name object as the corresponding group number.

        Input
        ========================

        {unamed_groups_json}

        """
        print(prompt)

        class NamedGroup(BaseModel):
            group_number: int
            group_headline: str


        class NamedGroups(BaseModel):
            named_groups: list[NamedGroup]

        reasoning_effort = "low"
        response = o1mini.run(
            prompt=prompt, response_model=NamedGroups, reasoning_effort=reasoning_effort
        )
        print(response)
        groups_number2group_headline = {named_group.group_number: named_group.group_headline for named_group in response.named_groups}
        named_clusters = {}
        for number, headline in groups_number2group_headline.items():
            named_clusters[headline] = [member for member in self.clusters[number]]
        self.named_clusters = named_clusters
        print(named_clusters)
        logger.warning(f"Reasoning effort: {reasoning_effort}")
        self.save()

    async def step3_classify_outliers_w_llm(
        self,
        start: int,
        size: Union[int, None],
        batch_size: int,
        llm_name: str,
        max_tokens: int,
        max_retries: int,
        timeout: Union[int, None],
        reasoning_effort: Optional[Literal["high", "medium", "low"]] = None,
    ) -> None:
        class Input(BaseModel):
            text: str

        labels = list(self.named_clusters.keys())
        class MultiClassPredictionResponse(BaseModel):

            tag: Optional[Literal[*labels]] = Field(
                default=None,
                title="Diet Tags",
                description="""
                    Tag the text with the best diet tag based on your scientific knowledge
                    of using changes in diet as an intervention to improve health outcomes.
                    If you are not sure leave it blank.
                """,
            )
        class MultiClassChain(Chain):

            input_schema = Input
            output_schema = MultiClassPredictionResponse

            @classmethod
            def make_input_text(cls, *, input: Input) -> str:
                input_text = f"""

                    {input.text}

                """
                return input_text
        if size is None:
            size = len(self.clusters[-1]) - start

        input_docs = self.clusters[-1][start : start + size]
        input_objects = [Input(text=doc.doc) for doc in input_docs]
        logger.info(f"Classifying {len(input_objects)} outliers")

        responses = await MultiClassChain.batch_predict(
            size=batch_size,
            llm_name=llm_name,
            input_objects=input_objects,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
        named_clusters_w_merged_outliers = self.named_clusters.copy()
        for response, input_object in zip(responses, input_objects):
            print(input_object.text, "=>", response.tag)
            # merge into the named cluster
            if response.tag is not None:
                named_clusters_w_merged_outliers[response.tag].append(LabeledDoc(doc=input_object.text, label=-1, llm_label=f"o3-mini:{reasoning_effort}"))
        self.named_clusters_w_merged_outliers = named_clusters_w_merged_outliers
        print(named_clusters_w_merged_outliers)
        self.save()
        return




if __name__ == "__main__":
    print("Running pipeline")
    source_path = "input_examples.json"
    with open(source_path, "r") as f:
        source_sentences = json.load(f)
    config = PipelineConfig(min_topic_size=4)
    pipeline = Pipeline(config=config)
    asyncio.run(pipeline.run(source_sentences=source_sentences))
