from __future__ import annotations

import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel
from rich import print
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({"info": "dim cyan", "warning": "magenta", "danger": "bold red"})
console = Console(theme=custom_theme)


class AzureOpenAIConfig(BaseModel):
    """Read + Write configs for Azure OpenAI API to JSON file to put in .env"""

    api_version: str
    azure_endpoint: str
    azure_deployment: str
    api_key: str
    timeout: int

    def to_json(self):
        """Developer helper tool to create a JSON file for manually adding to your local .env"""
        file_path = self.azure_deployment + "azure.json"
        with open(file_path, "w") as f:
            f.write(json.dumps(self.model_dump_json()))
        logger.success(f"Saved Azure OpenAI config to {file_path}")


class LabeledDoc(BaseModel):
    pos: Optional[int] = None
    doc: str
    label: int
    prob: Optional[float] = None
    llm_label: Optional[str] = None


class Clusters(BaseModel):
    clusters: dict[int, list[LabeledDoc]]
    bertopic_kwargs: dict
    embedding_llm_name: str

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(json.dumps(self.model_dump_json(indent=4)))
        logger.success(f"Saved clusters to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> Clusters:
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)
