import os

from dotenv import load_dotenv
from openai import OpenAI
from rich import print

from bertopic_easy.cluster import cluster
from bertopic_easy.naming import name

load_dotenv()

sentences = [
    "16/8 fasting",
    "16:8 fasting",
    "24-hour fasting",
    "24-hour one meal a day (OMAD) eating pattern",
    "2:1 ketogenic diet, low-glycemic-index diet",
    "30-day nutrition plan",
    "36-hour fast",
    "4-day fast",
    "40 hour fast, low carb meals",
    "4:3 fasting",
    "5-day fasting-mimicking diet (FMD) program",
    "7 day fast",
    "84-hour fast",
    "90/10 diet",
    "Adjusting macro and micro nutrient intake",
    "Adjusting target macros",
    "Macro and micro nutrient intake",
    "AllerPro formula",
    "Alternate Day Fasting (ADF), One Meal A Day (OMAD)",
    "American cheese",
    "Atkin's diet",
    "Atkins diet",
    "Avoid seed oils",
    "Avoiding seed oils",
    "Limiting seed oils",
    "Limited seed oils and processed foods",
    "Avoiding seed oils and processed foods",
]

openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

clusters = cluster(
    bertopic_kwargs=dict(min_topic_size=4),
    docs=sentences,
    openai=openai,
    embed_llm_name="text-embedding-3-large",
    with_disk_cache=True,
)


named_clusters = name(
    clusters=clusters,
    openai=openai,
    llm_model_name="o3-mini",
    reasoning_effort="low",
    subject="personal diet intervention outcomes",
)
print(named_clusters)
