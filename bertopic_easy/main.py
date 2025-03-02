import os
from typing import Literal

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from rich import print

from bertopic_easy.classify_outliers import classify_outliers
from bertopic_easy.cluster import cluster
from bertopic_easy.input_examples import diet_actions
from bertopic_easy.models import AzureOpenAIConfig, Clusters
from bertopic_easy.naming import name

load_dotenv()
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
async_openai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def bertopic_easy(
    *,
    texts: list[str],
    openai_api_key: str,
    reasoning_effort: Literal["low", "medium", "high"],
    subject: str,
) -> Clusters:
    openai = OpenAI(api_key=openai_api_key)
    async_openai = AsyncOpenAI(api_key=openai_api_key)
    clusters = cluster(
        bertopic_kwargs=dict(min_topic_size=4),
        docs=texts,
        openai=openai,
        embed_llm_name="text-embedding-3-large",
        with_disk_cache=True,
    )
    named_clusters = name(
        clusters=clusters,
        openai=openai,
        llm_model_name="o3-mini",
        reasoning_effort="low",
        subject=subject,
    )
    try:
        merged = classify_outliers(
            named_clusters=named_clusters,
            outliers=clusters.clusters[-1],
            openai=async_openai,
            llm_name="o3-mini",  # ONLY THIS LLM ALLOWED for now
            reasoning_effort=reasoning_effort,
        )
    except KeyError:
        logger.debug("No outliers found")
        return named_clusters
    return merged


def bertopic_easy_azure(
    *,
    texts: list[str],
    reasoning_effort: Literal["low", "medium", "high"],
    subject: str,
    azure_embeder_config: AzureOpenAIConfig,
    azure_namer_config: AzureOpenAIConfig,
    azure_classifier_config: AzureOpenAIConfig,
) -> Clusters:
    clusters = cluster(
        bertopic_kwargs=dict(min_topic_size=4),
        docs=texts,
        openai=AzureOpenAI(**azure_embeder_config.model_dump()),
        embed_llm_name=azure_embeder_config.azure_deployment,
        with_disk_cache=True,
    )
    named_clusters = name(
        clusters=clusters,
        openai=AzureOpenAI(**azure_namer_config.model_dump()),
        llm_model_name=azure_namer_config.azure_deployment,
        reasoning_effort=reasoning_effort,
        subject=subject,
    )
    try:
        merged = classify_outliers(
            named_clusters=named_clusters,
            outliers=clusters.clusters[-1],
            openai=AsyncAzureOpenAI(**azure_classifier_config.model_dump()),
            llm_name=azure_classifier_config.azure_deployment,  # ONLY THIS LLM ALLOWED for now
            reasoning_effort=reasoning_effort,
        )
    except KeyError:
        logger.debug("No outliers found")
        return named_clusters
    return merged
