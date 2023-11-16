"""Module for calculating distance matrix via Gower method."""
from typing import Hashable, List
import numpy as np
import pandas as pd
from deepchecks.utils.array_math import fast_sum_by_row

def gower_matrix(data: np.ndarray, cat_features: np.array) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    "\n    Calculate distance matrix for a dataset using Gower's method.\n\n    Gowers distance is a measurement for distance between two samples. It returns the average of their distances\n    per feature. For numeric features it calculates the absolute distance divide by the range of the feature. For\n    categorical features it is an indicator whether the values are the same.\n    See https://www.jstor.org/stable/2528823 for further details. In addition, it can deal with missing values.\n    Note that this method is expensive in memory and requires keeping in memory a matrix of size data*data.\n\n    Parameters\n    ----------\n    data: numpy.ndarray\n        Dataset matrix.\n    cat_features: numpy.array\n        Boolean array of representing which of the columns are categorical features.\n\n    Returns\n    -------\n    numpy.ndarray\n     representing the distance matrix.\n    "
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    feature_ranges = np.ones(data.shape[1]) * -1
    feature_ranges[~cat_features] = np.nanmax(data[:, ~cat_features], axis=0) - np.nanmin(data[:, ~cat_features], axis=0)
    result = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            value = calculate_distance(data[i, :], data[j, :], feature_ranges)
            result[i, j] = value
            result[j, i] = value
    return result

def calculate_nearest_neighbors_distances(data: pd.DataFrame, cat_cols: List[Hashable], numeric_cols: List[Hashable], num_neighbors: int, samples_to_calc_neighbors_for: pd.DataFrame=None):
    if False:
        i = 10
        return i + 15
    "\n    Calculate distance matrix for a dataset using Gower's method.\n\n    Gowers distance is a measurement for distance between two samples. It returns the average of their distances\n    per feature. For numeric features it calculates the absolute distance divide by the range of the feature. For\n    categorical features it is an indicator whether the values are the same.\n    See https://www.jstor.org/stable/2528823 for further details.\n    This method minimizes memory usage by saving in memory and returning only the closest neighbors of each sample.\n    In addition, it can deal with missing values.\n\n    Parameters\n    ----------\n    data: pd.DataFrame\n        DataFrame including all\n    cat_cols: List[Hashable]\n        List of categorical columns in the data.\n    numeric_cols: List[Hashable]\n        List of numerical columns in the data.\n    num_neighbors: int\n        Number of neighbors to return. For example, for n=2 for each sample returns the distances to the two closest\n        samples in the dataset.\n    samples_to_calc_neighbors_for: pd.DataFrame, default None\n        Samples for which to calculate nearest neighbors. If None, calculates for all given samples in data.\n        These samples do not have to exist in data, but must share all relevant features.\n\n    Returns\n    -------\n    numpy.ndarray\n        representing the distance matrix to the nearest neighbors.\n    numpy.ndarray\n        representing the indexes of the nearest neighbors.\n    "
    num_samples = data.shape[0]
    if samples_to_calc_neighbors_for is not None:
        data = pd.concat([data, samples_to_calc_neighbors_for])
        num_indices_to_calc = samples_to_calc_neighbors_for.shape[0]
    else:
        num_indices_to_calc = data.shape[0]
    cat_data = data[cat_cols]
    numeric_data = data[numeric_cols]
    num_features = len(cat_cols + numeric_cols)
    (distances, indexes) = (np.zeros((num_indices_to_calc, num_neighbors)), np.zeros((num_indices_to_calc, num_neighbors)))
    cat_data = np.asarray(cat_data.apply(lambda x: pd.factorize(x)[0])) if not cat_data.empty else np.asarray(cat_data)
    numeric_data = np.asarray(numeric_data.fillna(value=np.nan).astype('float64'))
    numeric_feature_ranges = np.nanmax(numeric_data, axis=0) - np.nanmin(numeric_data, axis=0)
    numeric_feature_ranges = np.where(numeric_feature_ranges == 0, 1, numeric_feature_ranges)
    numeric_data = np.nan_to_num(numeric_data, nan=np.inf)
    original_error_state = np.geterr()['invalid']
    np.seterr(invalid='ignore')
    if samples_to_calc_neighbors_for is not None:
        numeric_samples_to_calc_neighbors_for = numeric_data[num_samples:]
        cat_samples_to_calc_neighbors_for = cat_data[num_samples:]
        numeric_data = numeric_data[:num_samples]
        cat_data = cat_data[:num_samples]
    else:
        numeric_samples_to_calc_neighbors_for = numeric_data
        cat_samples_to_calc_neighbors_for = cat_data
    for i in range(num_indices_to_calc):
        numeric_sample_i = numeric_samples_to_calc_neighbors_for[i, :]
        cat_sample_i = cat_samples_to_calc_neighbors_for[i, :]
        dist_to_sample_i = _calculate_distances_to_sample(categorical_sample=cat_sample_i, numeric_sample=numeric_sample_i, cat_data=cat_data, numeric_data=numeric_data, numeric_feature_ranges=numeric_feature_ranges, num_features=num_features)
        min_dist_indexes = np.argpartition(dist_to_sample_i, num_neighbors)[:num_neighbors]
        min_dist_indexes_ordered = sorted(min_dist_indexes, key=lambda x, arr=dist_to_sample_i: arr[x], reverse=False)
        indexes[i, :] = min_dist_indexes_ordered
        distances[i, :] = dist_to_sample_i[min_dist_indexes_ordered]
    np.seterr(invalid=original_error_state)
    return (np.nan_to_num(distances, nan=np.nan, posinf=np.nan, neginf=np.nan), indexes)

def _calculate_distances_to_sample(categorical_sample: np.ndarray, numeric_sample: np.ndarray, cat_data: np.ndarray, numeric_data: np.ndarray, numeric_feature_ranges: np.ndarray, num_features: int):
    if False:
        for i in range(10):
            print('nop')
    "\n    Calculate Gower's distance between a single sample to the rest of the samples in the dataset.\n\n    Parameters\n    ----------\n    categorical_sample\n        The categorical features part of the sample to compare to the rest of the samples.\n    numeric_sample\n        The numeric features part of the sample to compare to the rest of the samples.\n    cat_data\n        The categorical features part of the dataset(after preprocessing).\n    numeric_data\n        The numeric features part of the dataset(after preprocessing).\n    numeric_feature_ranges\n        The range sizes of each numerical feature.\n    num_features\n        The total number of features in the dataset.\n    Returns\n    -------\n    numpy.ndarray\n        The distances to the rest of the samples.\n    "
    numeric_feat_dist_to_sample = numeric_data - numeric_sample
    np.abs(numeric_feat_dist_to_sample, out=numeric_feat_dist_to_sample)
    null_dist_locations = np.logical_or(numeric_feat_dist_to_sample == np.inf, numeric_feat_dist_to_sample == np.nan)
    null_numeric_features_per_sample = fast_sum_by_row(null_dist_locations)
    numeric_feat_dist_to_sample[null_dist_locations] = 0
    numeric_feat_dist_to_sample = numeric_feat_dist_to_sample.astype('float64')
    np.divide(numeric_feat_dist_to_sample, numeric_feature_ranges, out=numeric_feat_dist_to_sample)
    cat_feature_dist_to_sample = cat_data - categorical_sample != 0
    dist_to_sample = fast_sum_by_row(cat_feature_dist_to_sample) + fast_sum_by_row(numeric_feat_dist_to_sample)
    return dist_to_sample / (-null_numeric_features_per_sample + num_features)

def calculate_distance(vec1: np.array, vec2: np.array, range_per_feature: np.array) -> float:
    if False:
        return 10
    "Calculate distance between two vectors using Gower's method.\n\n    Parameters\n    ----------\n    vec1 : np.array\n        First vector.\n    vec2 : np.array\n        Second vector.\n    range_per_feature : np.array\n        Range of each numeric feature or -1 for categorical.\n\n    Returns\n    -------\n    float\n     representing Gower's distance between the two vectors.\n    "
    sum_dist = 0
    num_features = 0
    for col_index in range(len(vec1)):
        if range_per_feature[col_index] == -1:
            if pd.isnull(vec1[col_index]) and pd.isnull(vec2[col_index]):
                sum_dist += 0
            elif (pd.isnull(vec1[col_index]) or pd.isnull(vec2[col_index])) or vec1[col_index] != vec2[col_index]:
                sum_dist += 1
            num_features += 1
        else:
            if pd.isnull(vec1[col_index]) or pd.isnull(vec2[col_index]):
                continue
            sum_dist += np.abs(vec1[col_index] - vec2[col_index]) / range_per_feature[col_index]
            num_features += 1
    if num_features == 0:
        return np.nan
    return sum_dist / num_features