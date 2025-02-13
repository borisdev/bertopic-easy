import asyncio
from typing import Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from bertopic_easy.models import Clusters, LabeledDoc


async def _classify_outliers(
    named_clusters: Clusters,
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

    clusters = named_clusters.clusters
    labels = list(clusters.keys())
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
        size = len(clusters[-1]) - start

    input_docs = clusters[-1][start : start + size]
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
    named_clusters_w_merged_outliers = {}
    for response, input_object in zip(responses, input_objects):
        print(input_object.text, "=>", response.tag)
        # merge into the named cluster
        if response.tag is not None:
            named_clusters_w_merged_outliers[response.tag].append(LabeledDoc(doc=input_object.text, label=-1, llm_label=f"o3-mini:{reasoning_effort}"))
    return named_clusters_w_merged_outliers

def classify_outliers(
    named_clusters: Clusters,
    start: int,
    size: Union[int, None],
    batch_size: int,
    llm_name: str,
    max_tokens: int,
    max_retries: int,
    timeout: Union[int, None],
    reasoning_effort: Optional[Literal["high", "medium", "low"]] = None,
) -> None:
    return asyncio.run(
        _classify_outliers(
            named_clusters=named_clusters,
            start=start,
            size=size,
            batch_size=batch_size,
            llm_name=llm_name,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
    )
