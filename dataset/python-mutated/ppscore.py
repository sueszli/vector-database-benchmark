"""PPS (Predictive Power Score) module."""
import warnings
warnings.filterwarnings('ignore', message='The least populated class in y has only')
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype, is_timedelta64_dtype
from sklearn import preprocessing, tree
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from deepchecks.utils.typing import Hashable
NOT_SUPPORTED_ANYMORE = 'NOT_SUPPORTED_ANYMORE'
TO_BE_CALCULATED = -1

def _calculate_model_cv_score_(df, target, feature, task, cross_validation, random_seed, **kwargs):
    if False:
        return 10
    'Calculate the mean model score based on cross-validation.'
    metric = task['metric_key']
    model = task['model']
    df = df.sample(frac=1, random_state=random_seed, replace=False)
    if task['type'] == 'classification':
        label_encoder = preprocessing.LabelEncoder()
        df[target] = label_encoder.fit_transform(df[target])
        target_series = df[target]
    else:
        target_series = df[target]
    if _dtype_represents_categories(df[feature]):
        one_hot_encoder = preprocessing.OneHotEncoder()
        array = df[feature].__array__()
        sparse_matrix = one_hot_encoder.fit_transform(array.reshape(-1, 1))
        feature_input = sparse_matrix
    else:
        array = df[feature].values
        if not isinstance(array, np.ndarray):
            array = array.to_numpy()
        feature_input = array.reshape(-1, 1)
    scores = cross_val_score(model, feature_input, target_series, cv=cross_validation, scoring=metric)
    return scores.mean()

def _normalized_mae_score(model_mae, naive_mae):
    if False:
        print('Hello World!')
    'Normalize the model MAE score, given the baseline score.'
    if model_mae > naive_mae:
        return 0
    else:
        return 1 - model_mae / naive_mae

def _mae_normalizer(df, y, model_score, **kwargs):
    if False:
        i = 10
        return i + 15
    'In case of MAE, calculates the baseline score for y and derives the PPS.'
    df['naive'] = df[y].median()
    baseline_score = mean_absolute_error(df[y], df['naive'])
    ppscore = _normalized_mae_score(abs(model_score), baseline_score)
    return (ppscore, baseline_score)

def _normalized_f1_score(model_f1, baseline_f1):
    if False:
        for i in range(10):
            print('nop')
    'Normalize the model F1 score, given the baseline score.'
    if model_f1 < baseline_f1:
        return 0
    else:
        scale_range = 1.0 - baseline_f1
        f1_diff = model_f1 - baseline_f1
        return f1_diff / scale_range

def _f1_normalizer(df, y, model_score, random_seed):
    if False:
        return 10
    'In case of F1, calculates the baseline score for y and derives the PPS.'
    label_encoder = preprocessing.LabelEncoder()
    df['truth'] = label_encoder.fit_transform(df[y])
    df['most_common_value'] = df['truth'].value_counts().index[0]
    random = df['truth'].sample(frac=1, random_state=random_seed)
    baseline_score = max(f1_score(df['truth'], df['most_common_value'], average='weighted'), f1_score(df['truth'], random, average='weighted'))
    ppscore = _normalized_f1_score(model_score, baseline_score)
    return (ppscore, baseline_score)
VALID_CALCULATIONS = {'regression': {'type': 'regression', 'is_valid_score': True, 'model_score': TO_BE_CALCULATED, 'baseline_score': TO_BE_CALCULATED, 'ppscore': TO_BE_CALCULATED, 'metric_name': 'mean absolute error', 'metric_key': 'neg_mean_absolute_error', 'model': tree.DecisionTreeRegressor(), 'score_normalizer': _mae_normalizer}, 'classification': {'type': 'classification', 'is_valid_score': True, 'model_score': TO_BE_CALCULATED, 'baseline_score': TO_BE_CALCULATED, 'ppscore': TO_BE_CALCULATED, 'metric_name': 'weighted F1', 'metric_key': 'f1_weighted', 'model': tree.DecisionTreeClassifier(), 'score_normalizer': _f1_normalizer}, 'predict_itself': {'type': 'predict_itself', 'is_valid_score': True, 'model_score': 1, 'baseline_score': 0, 'ppscore': 1, 'metric_name': None, 'metric_key': None, 'model': None, 'score_normalizer': None}, 'target_is_constant': {'type': 'target_is_constant', 'is_valid_score': True, 'model_score': 1, 'baseline_score': 1, 'ppscore': 0, 'metric_name': None, 'metric_key': None, 'model': None, 'score_normalizer': None}, 'target_is_id': {'type': 'target_is_id', 'is_valid_score': True, 'model_score': 0, 'baseline_score': 0, 'ppscore': 0, 'metric_name': None, 'metric_key': None, 'model': None, 'score_normalizer': None}, 'feature_is_id': {'type': 'feature_is_id', 'is_valid_score': True, 'model_score': 0, 'baseline_score': 0, 'ppscore': 0, 'metric_name': None, 'metric_key': None, 'model': None, 'score_normalizer': None}}
INVALID_CALCULATIONS = ['target_is_datetime', 'target_data_type_not_supported', 'empty_dataframe_after_dropping_na', 'unknown_error']

def _dtype_represents_categories(series) -> bool:
    if False:
        while True:
            i = 10
    'Determine if the dtype of the series represents categorical values.'
    return is_bool_dtype(series) or is_object_dtype(series) or is_string_dtype(series) or is_categorical_dtype(series)

def _determine_case_and_prepare_df(df, x, y, sample=5000, random_seed=123):
    if False:
        while True:
            i = 10
    'Return str with the name of the determined case based on the columns x and y.'
    if x == y:
        return (df, 'predict_itself')
    df = df[[x, y]]
    df = df.dropna()
    if len(df) == 0:
        return (df, 'empty_dataframe_after_dropping_na')
    df = _maybe_sample(df, sample, random_seed=random_seed)
    if _feature_is_id(df, x):
        return (df, 'feature_is_id')
    category_count = df[y].value_counts().count()
    if category_count == 1:
        return (df, 'target_is_constant')
    if _dtype_represents_categories(df[y]) and category_count == len(df[y]):
        return (df, 'target_is_id')
    if _dtype_represents_categories(df[y]):
        return (df, 'classification')
    if is_numeric_dtype(df[y]):
        return (df, 'regression')
    if is_datetime64_any_dtype(df[y]) or is_timedelta64_dtype(df[y]):
        return (df, 'target_is_datetime')
    return (df, 'target_data_type_not_supported')

def _feature_is_id(df, x):
    if False:
        while True:
            i = 10
    'Return Boolean if the feature column x is an ID.'
    if not _dtype_represents_categories(df[x]):
        return False
    category_count = df[x].value_counts().count()
    return category_count == len(df[x])

def _maybe_sample(df, sample, random_seed=None):
    if False:
        i = 10
        return i + 15
    '\n    Maybe samples the rows of the given df to have at most `sample` rows.\n\n    If sample is `None` or falsy, there will be no sampling.\n    If the df has fewer rows than the sample, there will be no sampling.\n\n    Parameters\n    ----------\n    df : pandas.DataFrame\n        Dataframe that might be sampled\n    sample : int or `None`\n        Number of rows to be sampled\n    random_seed : int or `None`\n        Random seed that is forwarded to pandas.DataFrame.sample as `random_state`\n\n    Returns\n    -------\n    pandas.DataFrame\n        DataFrame after potential sampling\n    '
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=random_seed, replace=False)
    return df

def _is_column_in_df(column, df):
    if False:
        while True:
            i = 10
    try:
        return column in df.columns
    except:
        return False

def _score(df, x, y, task, sample, cross_validation, random_seed, invalid_score, catch_errors):
    if False:
        while True:
            i = 10
    (df, case_type) = _determine_case_and_prepare_df(df, x, y, sample=sample, random_seed=random_seed)
    task = _get_task(case_type, invalid_score)
    if case_type in ['classification', 'regression']:
        model_score = _calculate_model_cv_score_(df, target=y, feature=x, task=task, cross_validation=cross_validation, random_seed=random_seed)
        (ppscore, baseline_score) = task['score_normalizer'](df, y, model_score, random_seed=random_seed)
    else:
        model_score = task['model_score']
        baseline_score = task['baseline_score']
        ppscore = task['ppscore']
    return {'x': x, 'y': y, 'ppscore': ppscore, 'case': case_type, 'is_valid_score': task['is_valid_score'], 'metric': task['metric_name'], 'baseline_score': baseline_score, 'model_score': abs(model_score), 'model': task['model']}

def score(df, x, y, task=NOT_SUPPORTED_ANYMORE, sample=5000, cross_validation=4, random_seed=123, invalid_score=0, catch_errors=True):
    if False:
        i = 10
        return i + 15
    '\n    Calculate the Predictive Power Score (PPS) for "x predicts y".\n\n    The score always ranges from 0 to 1 and is data-type agnostic.\n\n    A score of 0 means that the column x cannot predict the column y better than a naive baseline model.\n    A score of 1 means that the column x can perfectly predict the column y given the model.\n    A score between 0 and 1 states the ratio of how much potential predictive power the model achieved compared to the\n    baseline model.\n\n    Parameters\n    ----------\n    df : pandas.DataFrame\n        Dataframe that contains the columns x and y\n    x : str\n        Name of the column x which acts as the feature\n    y : str\n        Name of the column y which acts as the target\n    sample : int or `None`\n        Number of rows for sampling. The sampling decreases the calculation time of the PPS.\n        If `None` there will be no sampling.\n    cross_validation : int\n        Number of iterations during cross-validation. This has the following implications:\n        For example, if the number is 4, then it is possible to detect patterns when there are at least 4 times the same\n         observation. If the limit is increased, the required minimum observations also increase. This is important,\n         because this is the limit when sklearn will throw an error and the PPS cannot be calculated\n    random_seed : int or `None`\n        Random seed for the parts of the calculation that require random numbers, e.g. shuffling or sampling.\n        If the value is set, the results will be reproducible. If the value is `None` a new random number is drawn at\n        the start of each calculation.\n    invalid_score : any\n        The score that is returned when a calculation is invalid, e.g. because the data type was not supported.\n    catch_errors : bool\n        If `True` all errors will be catched and reported as `unknown_error` which ensures convenience. If `False`\n        errors will be raised. This is helpful for inspecting and debugging errors.\n\n    Returns\n    -------\n    Dict\n        A dict that contains multiple fields about the resulting PPS.\n        The dict enables introspection into the calculations that have been performed under the hood\n    '
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame")
    if not _is_column_in_df(x, df):
        raise ValueError(f"The 'x' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe")
    if len(df[[x]].columns) >= 2:
        raise AssertionError(f'The dataframe has {len(df[[x]].columns)} columns with the same column name {x}\nPlease adjust the dataframe and make sure that only 1 column has the name {x}')
    if not _is_column_in_df(y, df):
        raise ValueError(f"The 'y' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe")
    if len(df[[y]].columns) >= 2:
        raise AssertionError(f'The dataframe has {len(df[[y]].columns)} columns with the same column name {y}\nPlease adjust the dataframe and make sure that only 1 column has the name {y}')
    if task is not NOT_SUPPORTED_ANYMORE:
        raise AttributeError("The attribute 'task' is no longer supported because it led to confusion and inconsistencies.\nThe task of the model is now determined based on the data types of the columns. If you want to change the task please adjust the data type of the column.\nFor more details, please refer to the README")
    if random_seed is None:
        from random import random
        random_seed = int(random() * 1000)
    try:
        return _score(df, x, y, task, sample, cross_validation, random_seed, invalid_score, catch_errors)
    except Exception as exception:
        if catch_errors:
            case_type = 'unknown_error'
            task = _get_task(case_type, invalid_score)
            return {'x': x, 'y': y, 'ppscore': task['ppscore'], 'case': case_type, 'is_valid_score': task['is_valid_score'], 'metric': task['metric_name'], 'baseline_score': task['baseline_score'], 'model_score': task['model_score'], 'model': task['model']}
        else:
            raise exception

def _get_task(case_type, invalid_score):
    if False:
        print('Hello World!')
    if case_type in VALID_CALCULATIONS.keys():
        return VALID_CALCULATIONS[case_type]
    elif case_type in INVALID_CALCULATIONS:
        return {'type': case_type, 'is_valid_score': False, 'model_score': invalid_score, 'baseline_score': invalid_score, 'ppscore': invalid_score, 'metric_name': None, 'metric_key': None, 'model': None, 'score_normalizer': None}
    raise Exception(f'case_type {case_type} is not supported')

def _format_list_of_dicts(scores, output, sorted):
    if False:
        while True:
            i = 10
    '\n    Format list of score dicts `scores`.\n\n    - maybe sort by ppscore\n    - maybe return pandas.Dataframe\n    - output can be one of ["df", "list"]\n    '
    if sorted:
        scores.sort(key=lambda item: item['ppscore'], reverse=True)
    if output == 'df':
        df_columns = ['x', 'y', 'ppscore', 'case', 'is_valid_score', 'metric', 'baseline_score', 'model_score', 'model']
        data = {column: [score[column] for score in scores] for column in df_columns}
        scores = pd.DataFrame.from_dict(data)
    return scores

def predictors(df, y: Hashable, output='df', sorted=True, **kwargs):
    if False:
        return 10
    '\n    Calculate the Predictive Power Score (PPS) of all the features in the dataframe.\n\n    against a target column\n\n    Parameters\n    ----------\n    df : pandas.DataFrame\n        The dataframe that contains the data\n    y : str\n        Name of the column y which acts as the target\n    output: str - potential values: "df", "list"\n        Control the type of the output. Either return a pandas.DataFrame (df) or a list with the score dicts\n    sorted: bool\n        Whether or not to sort the output dataframe/list by the ppscore\n    kwargs:\n        Other key-word arguments that shall be forwarded to the pps.score method,\n        e.g. `sample, `cross_validation, `random_seed, `invalid_score`, `catch_errors`\n\n    Returns\n    -------\n    pandas.DataFrame or list of Dict\n        Either returns a tidy dataframe or a list of all the PPS dicts. This can be influenced\n        by the output argument\n    '
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame")
    if not _is_column_in_df(y, df):
        raise ValueError(f"The 'y' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe")
    if len(df[[y]].columns) >= 2:
        raise AssertionError(f'The dataframe has {len(df[[y]].columns)} columns with the same column name {y}\nPlease adjust the dataframe and make sure that only 1 column has the name {y}')
    if not output in ['df', 'list']:
        raise ValueError(f"""The 'output' argument should be one of ["df", "list"] but you passed: {output}\nPlease adjust your input to one of the valid values""")
    if not sorted in [True, False]:
        raise ValueError(f"The 'sorted' argument should be one of [True, False] but you passed: {sorted}\nPlease adjust your input to one of the valid values")
    scores = [score(df, column, y, **kwargs) for column in df if column != y]
    return _format_list_of_dicts(scores=scores, output=output, sorted=sorted)

def matrix(df, output='df', sorted=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate the Predictive Power Score (PPS) matrix for all columns in the dataframe.\n\n    Args:\n        df : pandas.DataFrame\n            The dataframe that contains the data\n        output: str - potential values: "df", "list"\n            Control the type of the output. Either return a pandas.DataFrame (df) or a list with the score dicts\n        sorted: bool\n            Whether or not to sort the output dataframe/list by the ppscore\n        kwargs:\n            Other key-word arguments that shall be forwarded to the pps.score method,\n            e.g. `sample, `cross_validation, `random_seed, `invalid_score`, `catch_errors`\n\n    Returns:\n        pandas.DataFrame or list of Dict\n            Either returns a tidy dataframe or a list of all the PPS dicts. This can be influenced\n            by the output argument\n    '
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame")
    if not output in ['df', 'list']:
        raise ValueError(f"""The 'output' argument should be one of ["df", "list"] but you passed: {output}\nPlease adjust your input to one of the valid values""")
    if not sorted in [True, False]:
        raise ValueError(f"The 'sorted' argument should be one of [True, False] but you passed: {sorted}\nPlease adjust your input to one of the valid values")
    scores = [score(df, x, y, **kwargs) for x in df for y in df]
    return _format_list_of_dicts(scores=scores, output=output, sorted=sorted)