"""Utils module for auto-detecting interesting segments in text."""
import warnings
from typing import Hashable, List, Optional, Union
import numpy as np
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksProcessError
from deepchecks.nlp import TextData
from deepchecks.utils.dataframes import select_from_dataframe

def get_relevant_data_table(text_data: TextData, data_type: str, columns: Union[Hashable, List[Hashable], None], ignore_columns: Union[Hashable, List[Hashable], None], n_top_features: Optional[int]):
    if False:
        while True:
            i = 10
    'Get relevant data table from the database.'
    if data_type == 'metadata':
        relevant_metadata = text_data.metadata[text_data.categorical_metadata + text_data.numerical_metadata]
        features = select_from_dataframe(relevant_metadata, columns, ignore_columns)
        cat_features = [col for col in features.columns if col in text_data.categorical_metadata]
    elif data_type == 'properties':
        features = select_from_dataframe(text_data.properties, columns, ignore_columns)
        cat_features = [col for col in features.columns if col in text_data.categorical_properties]
    else:
        raise DeepchecksProcessError(f'Unknown segment_by value: {data_type}')
    if features.shape[1] < 2:
        raise DeepchecksNotSupportedError(f'Check requires to have at least two {data_type} columns in order to run.')
    if n_top_features is not None and n_top_features < features.shape[1]:
        _warn_n_top_columns(data_type, n_top_features)
        features = features.iloc[:, np.random.choice(features.shape[1], n_top_features, replace=False)]
    return (features, cat_features)

def _warn_n_top_columns(data_type: str, n_top_features: int):
    if False:
        return 10
    'Warn if n_top_columns is smaller than the number of segmenting features (metadata or properties).'
    if data_type == 'metadata':
        features_name = 'metadata columns'
        n_top_columns_parameter = 'n_top_columns'
        columns_parameter = 'columns'
    else:
        features_name = 'properties'
        n_top_columns_parameter = 'n_top_properties'
        columns_parameter = 'properties'
    warnings.warn(f'Parameter {n_top_columns_parameter} is set to {n_top_features} to avoid long computation time. This means that the check will run on {n_top_features} {features_name} selected at random. If you want to run on all {features_name}, set {n_top_columns_parameter} to None. Alternatively, you can set parameter {columns_parameter} to a list of the specific {features_name} you want to run on.', UserWarning)