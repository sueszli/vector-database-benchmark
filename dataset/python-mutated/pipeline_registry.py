"""Project pipelines."""
from __future__ import annotations
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

def register_pipelines() -> dict[str, Pipeline]:
    if False:
        while True:
            i = 10
    "Register the project's pipelines.\n\n    Returns:\n        A mapping from pipeline names to ``Pipeline`` objects.\n    "
    pipelines = find_pipelines()
    pipelines['__default__'] = sum(pipelines.values())
    return pipelines