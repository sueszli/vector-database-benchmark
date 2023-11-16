"""Compute statistical description of datasets."""
from typing import Any
from multimethod import multimethod
from tqdm import tqdm
from visions import VisionsTypeset
from ydata_profiling.config import Settings
from ydata_profiling.model.summarizer import BaseSummarizer

@multimethod
def describe_1d(config: Settings, series: Any, summarizer: BaseSummarizer, typeset: VisionsTypeset) -> dict:
    if False:
        while True:
            i = 10
    raise NotImplementedError()

@multimethod
def get_series_descriptions(config: Settings, df: Any, summarizer: BaseSummarizer, typeset: VisionsTypeset, pbar: tqdm) -> dict:
    if False:
        print('Hello World!')
    raise NotImplementedError()