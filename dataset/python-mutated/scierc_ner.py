"""Dataset containing abstracts from scientific journals and named entity annotations for relevant scientific words.

The data contains 350 samples for the train set and 100 samples for the test set. Each dataset sample is a tokenized
abstract from a scientific journal. Each token is annotated with a named entity tag. The dataset contains 7 named
entity tags: Task, Method, Material, Metric, OtherScientificTerm, and Generic. The dataset is a subset of
the SciERC dataset (http://nlp.cs.washington.edu/sciIE/)

Original publication:
Luan, Yi, He, Luheng, Ostendorf, Mari, and Hajishirzi, Hannaneh. (2018). "Multi-Task
Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction." In Proceedings
of the Conference on Empirical Methods in Natural Language Processing (EMNLP).

The SCIERC dataset, in turn, was extracted from the S2ORC dataset (https://github.com/allenai/s2orc).

Citation for the S2ORC dataset: Lo, Kyle, Wang, Lucy Lu, Neumann, Mark, Kinney, Rodney, and Weld, Daniel. (2020).
"S2ORC: The Semantic Scholar Open Research Corpus." In Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics. Online. Association for Computational Linguistics.
https://www.aclweb.org/anthology/2020.acl-main.447. doi: 10.18653/v1/2020.acl-main.447. pp. 4969-4983.

The S2ORC dataset is licensed under the ODC-By 1.0 licence (https://opendatacommons.org/licenses/by/1-0/) by the
AllanAI institute
"""
import pathlib
import typing as t
import warnings
import numpy as np
import pandas as pd
from deepchecks.nlp import TextData
from deepchecks.utils.builtin_datasets_utils import read_and_save_data
__all__ = ['load_data']
_DATA_JSON_URL = 'https://figshare.com/ndownloader/files/40878617'
_TRAIN_PROP = 'https://figshare.com/ndownloader/files/40878629'
_TEST_PROP = 'https://figshare.com/ndownloader/files/40878623'
_TRAIN_EMBEDDINGS_URL = 'https://figshare.com/ndownloader/files/40878626'
_TEST_EMBEDDINGS_URL = 'https://figshare.com/ndownloader/files/40878620'
ASSETS_DIR = pathlib.Path(__file__).absolute().parent.parent / 'assets' / 'scierc'

def load_all_data() -> t.Dict[str, t.Dict[str, t.Any]]:
    if False:
        return 10
    "Load a dict of all the text data, labels and predictions. One function because it's very lightweight."
    return read_and_save_data(ASSETS_DIR, 'scierc_data_dict.json', _DATA_JSON_URL, file_type='json')

def load_precalculated_predictions() -> t.Tuple[t.List[str], t.List[str]]:
    if False:
        i = 10
        return i + 15
    'Load and return a precalculated predictions for the dataset.\n\n    Returns\n    -------\n    predictions : Tuple[List[str], List[str]]\n        The IOB predictions of the tokens in the train and test datasets.\n    '
    data_dict = load_all_data()
    return (data_dict['train']['pred'], data_dict['test']['pred'])

def load_embeddings() -> t.Tuple[np.array, np.array]:
    if False:
        for i in range(10):
            print('nop')
    'Load and return the embeddings of the SCIERC dataset calculated by OpenAI.\n\n    Returns\n    -------\n    embeddings : np.Tuple[np.array, np.array]\n        Embeddings for the SCIERC dataset.\n    '
    train_embeddings = read_and_save_data(ASSETS_DIR, 'train_embeddings.npy', _TRAIN_EMBEDDINGS_URL, file_type='npy', to_numpy=True)
    test_embeddings = read_and_save_data(ASSETS_DIR, 'test_embeddings.npy', _TEST_EMBEDDINGS_URL, file_type='npy', to_numpy=True)
    return (train_embeddings, test_embeddings)

def load_properties() -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    if False:
        for i in range(10):
            print('nop')
    'Load and return the properties of the SCIERC dataset.\n\n    Returns\n    -------\n    properties : Tuple[pd.DataFrame, pd.DataFrame]\n        Properties for the SCIERC dataset.\n    '
    train_properties = read_and_save_data(ASSETS_DIR, 'train_properties.csv', _TRAIN_PROP, to_numpy=False, include_index=False)
    test_properties = read_and_save_data(ASSETS_DIR, 'test_properties.csv', _TEST_PROP, to_numpy=False, include_index=False)
    return (train_properties, test_properties)

def load_data(data_format: str='TextData', include_properties: bool=True, include_embeddings: bool=False) -> t.Tuple[t.Union[TextData, pd.DataFrame], t.Union[TextData, pd.DataFrame]]:
    if False:
        for i in range(10):
            print('nop')
    "Load and returns the SCIERC Abstract NER dataset (token classification).\n\n    Parameters\n    ----------\n    data_format : str, default: 'TextData'\n        Represent the format of the returned value. Can be 'TextData'|'Dict'\n        'TextData' will return the data as a TextData object\n        'Dict' will return the data as a dict of tokenized texts and IOB NER labels\n    include_properties : bool, default: True\n        If True, the returned data will include properties of the comments. Incompatible with data_format='DataFrame'\n    include_embeddings : bool, default: False\n        If True, the returned data will include embeddings of the comments. Incompatible with data_format='DataFrame'\n\n    Returns\n    -------\n    train, test : Tuple[Union[TextData, Dict]\n        Tuple of two objects represents the dataset split to train and test sets.\n    "
    if data_format.lower() not in ['textdata', 'dict']:
        raise ValueError('data_format must be either "TextData" or "Dict"')
    elif data_format.lower() == 'dict':
        if include_properties or include_embeddings:
            warnings.warn('include_properties and include_embeddings are incompatible with data_format="Dict". loading only original text data', UserWarning)
            (include_properties, include_embeddings) = (False, False)
    data = load_all_data()
    (train, test) = (data['train'], data['test'])
    del train['pred']
    del test['pred']
    if data_format.lower() != 'textdata':
        return (train, test)
    if include_properties:
        (train_properties, test_properties) = load_properties()
    else:
        (train_properties, test_properties) = (None, None)
    if include_embeddings:
        (train_embeddings, test_embeddings) = load_embeddings()
    else:
        (train_embeddings, test_embeddings) = (None, None)
    train_ds = TextData(tokenized_text=train['text'], label=train['text'], task_type='token_classification', properties=train_properties, embeddings=train_embeddings)
    test_ds = TextData(tokenized_text=test['text'], label=test['text'], task_type='token_classification', properties=test_properties, embeddings=test_embeddings)
    return (train_ds, test_ds)