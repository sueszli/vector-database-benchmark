"""
Module which implements feature PCA compression and PCA analysis of feature importance.
"""
import pandas as pd
import numpy as np
from scipy.stats import weightedtau, kendalltau, spearmanr, pearsonr

def _get_eigen_vector(dot_matrix, variance_thresh, num_features=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Advances in Financial Machine Learning, Snippet 8.5, page 119.\n\n    Computation of Orthogonal Features\n\n    Gets eigen values and eigen vector from matrix which explain % variance_thresh of total variance.\n\n    :param dot_matrix: (np.array): Matrix for which eigen values/vectors should be computed.\n    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain.\n    :param num_features: (int) Manually set number of features, overrides variance_thresh. (None by default)\n    :return: (pd.Series, pd.DataFrame): Eigenvalues, Eigenvectors.\n    '
    pass

def _standardize_df(data_frame):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function which divides df by std and extracts mean.\n\n    :param data_frame: (pd.DataFrame): Dataframe to standardize\n    :return: (pd.DataFrame): Standardized dataframe\n    '
    pass

def get_orthogonal_features(feature_df, variance_thresh=0.95, num_features=None):
    if False:
        print('Hello World!')
    '\n    Advances in Financial Machine Learning, Snippet 8.5, page 119.\n\n    Computation of Orthogonal Features.\n\n    Gets PCA orthogonal features.\n\n    :param feature_df: (pd.DataFrame): Dataframe of features.\n    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain.\n    :param num_features: (int) Manually set number of features, overrides variance_thresh. (None by default)\n    :return: (pd.DataFrame): Compressed PCA features which explain %variance_thresh of variance.\n    '
    pass

def get_pca_rank_weighted_kendall_tau(feature_imp, pca_rank):
    if False:
        return 10
    "\n    Advances in Financial Machine Learning, Snippet 8.6, page 121.\n\n    Computes Weighted Kendall's Tau Between Feature Importance and Inverse PCA Ranking.\n\n    :param feature_imp: (np.array): Feature mean importance.\n    :param pca_rank: (np.array): PCA based feature importance rank.\n    :return: (float): Weighted Kendall Tau of feature importance and inverse PCA rank with p_value.\n    "
    pass

def feature_pca_analysis(feature_df, feature_importance, variance_thresh=0.95):
    if False:
        for i in range(10):
            print('nop')
    '\n    Performs correlation analysis between feature importance (MDI for example, supervised) and PCA eigenvalues\n    (unsupervised).\n\n    High correlation means that probably the pattern identified by the ML algorithm is not entirely overfit.\n\n    :param feature_df: (pd.DataFrame): Features dataframe.\n    :param feature_importance: (pd.DataFrame): Individual MDI feature importance.\n    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain in PCA compression.\n    :return: (dict): Dictionary with kendall, spearman, pearson and weighted_kendall correlations and p_values.\n    '
    pass