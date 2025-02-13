import json

from bertopic_easy.pipeline import Pipeline, PipelineConfig


def test_pipeline_run():

    source_path = "input_examples.json"
    with open(source_path, "r") as f:
        source_sentences = json.load(f)
    config = PipelineConfig(min_topic_size=4)
    pipeline = Pipeline(config=config)
    pipeline.run(source_sentences=source_sentences)
