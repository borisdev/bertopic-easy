import json
import os

import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from rich import print

from bertopic_easy import bertopic_easy
from bertopic_easy.classify_outliers import classify_outliers
from bertopic_easy.cluster import cluster
from bertopic_easy.embedding import embed
from bertopic_easy.input_examples import diet_actions
from bertopic_easy.naming import name

load_dotenv()
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
async_openai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
USE_AZURE = False
## AZURE OPENAI CONFIG ##
# from openai import AzureOpenAI
# from bertopic_easy.models import AzureOpenAIConfig
# azure_openai_json = os.environ.get("text-embedding-3-large")
# if azure_openai_json is None:
#     raise ValueError(
#         "add the AzureOpenAI's `text-embedding-3-large` config to .env file"
#     )
# azure_openai_config = AzureOpenAIConfig(**json.loads(azure_openai_json))
#
# embedding_client = AzureOpenAI(
#     api_version=azure_openai_config.api_version,
#     azure_endpoint=azure_openai_config.azure_endpoint,
#     azure_deployment=azure_openai_config.azure_deployment,  # model name
#     api_key=azure_openai_config.api_key,
#     timeout=azure_openai_config.timeout,
# )
# classifier_client = AzureOpenAI(
#     api_version="2024-12-01-preview",
#     azure_endpoint="https://boris-m3ndov9n-eastus2.openai.azure.com/",
#     azure_deployment="o3-mini",  # model name
#     api_key=azure_openai_config.api_key,
# )


def test_embeddings():
    if USE_AZURE:
        raise NotImplementedError("Azure OpenAI not implemented")
    else:
        embedding_client = openai
        llm_model_name = "text-embedding-3-large"

    sample_size = 20
    embeddings = embed(
        texts=diet_actions[:sample_size],
        openai=embedding_client,
        llm_model_name=llm_model_name,
        with_disk_cache=False,
    )
    print(embeddings)


def test_embeddings_w_cache():
    if USE_AZURE:
        raise NotImplementedError("Azure OpenAI not implemented")
    else:
        embedding_client = openai
        llm_model_name = "text-embedding-3-large"
    sample_size = 20
    embeddings = embed(
        texts=diet_actions[:sample_size],
        openai=embedding_client,
        llm_model_name=llm_model_name,
        with_disk_cache=True,
    )
    print(embeddings)


@pytest.fixture(scope="session")
def test_clusters():
    if USE_AZURE:
        raise NotImplementedError("Azure OpenAI not implemented")
    else:
        embedding_client = openai
        llm_model_name = "text-embedding-3-large"
    clusters = cluster(
        bertopic_kwargs=dict(min_topic_size=4),
        docs=diet_actions,
        openai=embedding_client,
        embed_llm_name=llm_model_name,
        with_disk_cache=True,
    )
    return clusters


@pytest.fixture(scope="session")
def test_cluster_naming(test_clusters):
    if USE_AZURE:
        raise NotImplementedError("Azure OpenAI not implemented")
    else:
        classifier_client = openai
    named_clusters = name(
        clusters=test_clusters,
        openai=classifier_client,
        llm_model_name="o3-mini",
        reasoning_effort="low",
        subject="personal diet intervention outcomes",
    )
    return named_clusters


def test_classify_outliers(test_cluster_naming, test_clusters):
    if USE_AZURE:
        raise NotImplementedError("Azure OpenAI not implemented")
    else:
        classifier_client = async_openai
    merged = classify_outliers(
        named_clusters=test_cluster_naming,
        outliers=test_clusters.clusters[-1],
        openai=classifier_client,
        llm_name="o3-mini",  # ONLY ONE ALLOWED
        reasoning_effort="low",
    )
    print(merged)


def test_bertopic_easy():
    if USE_AZURE:
        raise NotImplementedError("Azure OpenAI not implemented")
    else:
        openai_api_key = os.environ["OPENAI_API_KEY"]
    clusters = bertopic_easy(
        texts=diet_actions,
        openai_api_key=openai_api_key,
        reasoning_effort="low",
        subject="personal diet intervention outcomes",
    )
    print(clusters)
