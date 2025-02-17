import os
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from rich import print

from bertopic_easy.classify_outliers import classify_outliers
from bertopic_easy.cluster import cluster
from bertopic_easy.input_examples import diet_actions
from bertopic_easy.models import Clusters
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
    merged = classify_outliers(
        named_clusters=named_clusters,
        outliers=clusters.clusters[-1],
        openai=async_openai,
        llm_name="o3-mini",  # ONLY THIS LLM ALLOWED for now
        reasoning_effort=reasoning_effort,
    )
    return merged
