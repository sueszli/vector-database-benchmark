"""Common utilities for model error analysis."""
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from deepchecks import tabular
from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.tabular import Dataset
from deepchecks.tabular.utils.feature_importance import _calculate_feature_importance
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.plot import colors
from deepchecks.utils.strings import format_number, format_percent

def model_error_contribution(train_dataset: pd.DataFrame, train_scores: pd.Series, test_dataset: pd.DataFrame, test_scores: pd.Series, numeric_features: List, categorical_features: List, min_error_model_score=0.5, random_state=42) -> Tuple[pd.Series, pd.Series]:
    if False:
        for i in range(10):
            print('nop')
    'Calculate features contributing to model error.'
    (error_model, new_feature_order) = create_error_regression_model(numeric_features, categorical_features, random_state=random_state)
    error_model.fit(train_dataset, y=train_scores)
    error_model_predicted = error_model.predict(test_dataset)
    error_model_score = r2_score(test_scores, error_model_predicted)
    if error_model_score < min_error_model_score:
        raise DeepchecksProcessError(f'Unable to train meaningful error model (r^2 score: {format_number(error_model_score)})')
    (error_fi, _) = _calculate_feature_importance(error_model, Dataset(test_dataset, test_scores, cat_features=categorical_features), model_classes=None, observed_classes=None, task_type=TaskType.REGRESSION, permutation_kwargs={'random_state': random_state, 'skip_messages': True})
    error_fi.index = new_feature_order
    error_fi.sort_values(ascending=False, inplace=True)
    return (error_fi, error_model_predicted)

def create_error_regression_model(numeric_features, cat_features, random_state=42) -> Tuple[Pipeline, List[Hashable]]:
    if False:
        i = 10
        return i + 15
    'Create regression model to calculate error.'
    numeric_transformer = SimpleImputer()
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', TargetEncoder())])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, cat_features)])
    return (Pipeline(steps=[('preprocessing', preprocessor), ('model', RandomForestRegressor(max_depth=4, n_jobs=-1, random_state=random_state))]), numeric_features + cat_features)

def error_model_display_dataframe(error_fi: pd.Series, error_model_predicted: pd.Series, dataset: pd.DataFrame, cat_features: List, max_features_to_show: int, min_feature_contribution: float, n_display_samples: int, min_segment_size: float, random_state: int, with_display: bool):
    if False:
        while True:
            i = 10
    'Wrap dataframe with tabular.Dataset for error_model_display with no scorer.'
    return error_model_display(error_fi, error_model_predicted, tabular.Dataset(dataset, cat_features=cat_features), None, None, max_features_to_show, min_feature_contribution, n_display_samples, min_segment_size, random_state, with_display)

def error_model_display(error_fi: pd.Series, error_model_predicted: pd.Series, dataset: tabular.Dataset, model: Optional[Any], scorer: Optional[Callable], max_features_to_show: int, min_feature_contribution: float, n_display_samples: int, min_segment_size: float, random_state: int, with_display: bool) -> Tuple[List, Dict]:
    if False:
        while True:
            i = 10
    'Calculate and display segments with large error discrepancies.\n\n    Parameters\n    ----------\n    error_fi : pd.Series\n        Feature Importances of the error model\n    error_model_predicted : pd.Series\n        Predictions of the values of the error model\n    dataset : tabular.Dataset\n        Dataset to create display from\n    model : Optional[Any]\n        Original model for calculating the score on tabular data (Must come with scorer)\n    scorer : Optional[Callable]\n        Scorer to calculate the output values of the segments (Must come with model)\n    max_features_to_show : int\n        Maximum number of features to output.\n    min_feature_contribution : float\n        Minimum value to consider a feature to output.\n    n_display_samples : int\n        Maximum number of values to represent in the display\n    min_segment_size : float\n        Minimum segment size to consider.\n    random_state: int\n        Random seed\n\n    Returns\n    -------\n    Tuple[List, Dict]:\n        List of display elements and Dict of segment description\n    '
    n_samples_display = min(n_display_samples, len(dataset))
    error_col_name = 'Deepchecks model error'
    display_error = pd.Series(error_model_predicted, name=error_col_name, index=dataset.data.index)
    display = []
    value = {'scorer_name': scorer.name if scorer else None, 'feature_segments': {}}
    weak_color = '#d74949'
    ok_color = colors['Test']
    for feature in error_fi.keys()[:max_features_to_show]:
        if error_fi[feature] < min_feature_contribution:
            break
        data = pd.concat([dataset.data[feature], display_error], axis=1)
        value['feature_segments'][feature] = {}
        segment1_details = {}
        segment2_details = {}
        if feature in dataset.cat_features:
            error_per_segment_ser = data.groupby(feature).agg(['mean', 'count'])[error_col_name].sort_values('mean', ascending=not scorer)
            cum_sum_ratio = error_per_segment_ser['count'].cumsum() / error_per_segment_ser['count'].sum()
            first_weakest_category_to_pass_min_segment_size = np.where(cum_sum_ratio.values >= min_segment_size)[0][0]
            in_segment_indices = np.arange(len(cum_sum_ratio)) <= first_weakest_category_to_pass_min_segment_size
            weak_categories = error_per_segment_ser.index[in_segment_indices]
            ok_categories = error_per_segment_ser.index[~in_segment_indices]
            if scorer:
                (ok_name_feature, segment1_details) = get_segment_details(model, scorer, dataset, data[feature].isin(ok_categories))
            else:
                (ok_name_feature, segment1_details) = get_segment_details_using_error(error_col_name, data, data[feature].isin(ok_categories))
            if with_display:
                color_map = {ok_name_feature: ok_color}
                if len(weak_categories) >= 1:
                    if scorer:
                        (weak_name_feature, segment2_details) = get_segment_details(model, scorer, dataset, data[feature].isin(weak_categories))
                    else:
                        (weak_name_feature, segment1_details) = get_segment_details_using_error(error_col_name, data, data[feature].isin(weak_categories))
                    color_map[weak_name_feature] = weak_color
                else:
                    weak_name_feature = None
                replace_dict = {x: weak_name_feature if x in weak_categories else ok_name_feature for x in error_per_segment_ser.index}
                color_col = data[feature].replace(replace_dict)
                display.append(px.violin(data, y=error_col_name, x=feature, title=f'Segmentation of error by feature: {feature}', box=False, labels={error_col_name: 'model error', 'color': 'Weak & OK Segments'}, color=color_col, color_discrete_map=color_map))
        elif feature in dataset.numerical_features:
            np.random.seed(random_state)
            sampling_idx = np.random.choice(range(len(data)), size=n_samples_display, replace=False)
            data = data.iloc[sampling_idx]
            tree_partitioner = DecisionTreeRegressor(max_depth=1, min_samples_leaf=min_segment_size + np.finfo(float).eps, random_state=random_state).fit(data[[feature]], data[error_col_name])
            if len(tree_partitioner.tree_.threshold) > 1:
                threshold = tree_partitioner.tree_.threshold[0]
                color_col = data[feature].ge(threshold)
                sampled_dataset = dataset.data.iloc[sampling_idx]
                if scorer:
                    (segment1_text, segment1_details) = get_segment_details(model, scorer, dataset.copy(sampled_dataset), color_col)
                    (segment2_text, segment2_details) = get_segment_details(model, scorer, dataset.copy(sampled_dataset), ~color_col)
                    segment1_ok = segment1_details['score'] >= segment2_details['score']
                    color_col = color_col.replace([True, False], [segment1_text, segment2_text])
                else:
                    (segment1_text, segment1_details) = get_segment_details_using_error(error_col_name, data, ~color_col)
                    (segment2_text, segment2_details) = get_segment_details_using_error(error_col_name, data, color_col)
                    segment1_ok = segment1_details['score'] < segment2_details['score']
                    color_col = color_col.replace([False, True], [segment1_text, segment2_text])
                if segment1_ok:
                    color_map = {segment1_text: ok_color, segment2_text: weak_color}
                    category_order = [segment2_text, segment1_text]
                else:
                    color_map = {segment1_text: weak_color, segment2_text: ok_color}
                    category_order = [segment1_text, segment2_text]
            else:
                color_col = data[error_col_name]
                color_map = None
                category_order = None
            if with_display:
                display.append(px.scatter(data, x=feature, y=error_col_name, color=color_col, title=f'Segmentation of error by the feature: {feature}', labels={error_col_name: 'model error', 'color': 'Weak & OK Segments'}, category_orders={'color': category_order}, color_discrete_map=color_map))
        if segment1_details:
            value['feature_segments'][feature]['segment1'] = segment1_details
        if segment2_details:
            value['feature_segments'][feature]['segment2'] = segment2_details
    return (display if with_display else None, value)

def get_segment_details(model: Any, scorer: Callable, dataset: tabular.Dataset, segment_condition_col: pd.Series) -> Tuple[str, Dict[str, float]]:
    if False:
        print('Hello World!')
    'Return details about the data segment using the scorer and model.'
    performance = scorer(model, dataset.copy(dataset.data[segment_condition_col.values]))
    n_samples = dataset.data[segment_condition_col].shape[0]
    segment_label = f'{scorer.name}: {format_number(performance)}, Samples: {n_samples} ({format_percent(n_samples / len(dataset))})'
    segment_details = {'score': performance, 'n_samples': n_samples, 'frac_samples': n_samples / len(dataset)}
    return (segment_label, segment_details)

def get_segment_details_using_error(error_column_name, dataset: pd.DataFrame, segment_condition_col: pd.Series) -> Tuple[str, Dict[str, float]]:
    if False:
        for i in range(10):
            print('nop')
    'Return details about the data segment using the error column.'
    n_samples = dataset[segment_condition_col].shape[0]
    performance = dataset[segment_condition_col][error_column_name].sum() / n_samples
    segment_label = f'Error: {format_number(performance)}, ({n_samples} samples - {format_percent(n_samples / len(dataset))} of the dataset)'
    segment_details = {'score': performance, 'n_samples': n_samples, 'frac_samples': n_samples / len(dataset)}
    return (segment_label, segment_details)