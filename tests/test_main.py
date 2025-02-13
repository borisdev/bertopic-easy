import json
import os

import pytest
from dotenv import load_dotenv
from openai import AzureOpenAI
from rich import print

from bertopic_easy.cluster import cluster
from bertopic_easy.embedding import embed
from bertopic_easy.input_examples import diet_actions
from bertopic_easy.models import AzureOpenAIConfig
from bertopic_easy.naming import name

load_dotenv()
azure_openai_json = os.environ.get("text-embedding-3-large")
if azure_openai_json is None:
    raise ValueError(
        "add the AzureOpenAI's `text-embedding-3-large` config to .env file"
    )
azure_openai_config = AzureOpenAIConfig(**json.loads(azure_openai_json))

embedding_client = AzureOpenAI(
    api_version=azure_openai_config.api_version,
    azure_endpoint=azure_openai_config.azure_endpoint,
    azure_deployment=azure_openai_config.azure_deployment,  # model name
    api_key=azure_openai_config.api_key,
    timeout=azure_openai_config.timeout,
)
classifier_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://boris-m3ndov9n-eastus2.openai.azure.com/",
    azure_deployment="o3-mini",  # model name
    api_key=azure_openai_config.api_key,
)


def test_embeddings():

    sample_size = 20
    embeddings = embed(
        texts=diet_actions[:sample_size],
        client=embedding_client,
        llm_model_name=azure_openai_config.azure_deployment,
        with_disk_cache=False,
    )
    print(embeddings)


def test_embeddings_w_cache():
    sample_size = 20
    embeddings = embed(
        texts=diet_actions[:sample_size],
        client=embedding_client,
        llm_model_name=azure_openai_config.azure_deployment,
        with_disk_cache=True,
    )
    print(embeddings)


@pytest.fixture(scope="session")
def test_clusters():
    clusters = cluster(
        bertopic_kwargs=dict(min_topic_size=4),
        docs=diet_actions,
        embedding_llm_client=embedding_client,
        embed_llm_name=azure_openai_config.azure_deployment,
        with_disk_cache=True,
    )
    return clusters


def test_cluster_naming(test_clusters):
    named_clusters = name(
        clusters=test_clusters,
        client=classifier_client,
        llm_model_name="o3-mini",
        reasoning_effort="low",
        subject="personal diet intervention outcomes",
    )
    print(named_clusters)


#
#
#
# @pytest.mark.asyncio
# async def test_pipeline_run():
#
#     config = PipelineConfig(min_topic_size=4)
#     pipeline = Pipeline(bertopic_kwargs=config, llm_config=None, request_config=None)
#     await pipeline.run(source_sentences=diet_actions)
