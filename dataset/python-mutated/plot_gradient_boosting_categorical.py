"""
================================================
Categorical Feature Support in Gradient Boosting
================================================

.. currentmodule:: sklearn

In this example, we will compare the training times and prediction
performances of :class:`~ensemble.HistGradientBoostingRegressor` with
different encoding strategies for categorical features. In
particular, we will evaluate:

- dropping the categorical features
- using a :class:`~preprocessing.OneHotEncoder`
- using an :class:`~preprocessing.OrdinalEncoder` and treat categories as
  ordered, equidistant quantities
- using an :class:`~preprocessing.OrdinalEncoder` and rely on the :ref:`native
  category support <categorical_support_gbdt>` of the
  :class:`~ensemble.HistGradientBoostingRegressor` estimator.

We will work with the Ames Lowa Housing dataset which consists of numerical
and categorical features, where the houses' sales prices is the target.

"""
from sklearn.datasets import fetch_openml
(X, y) = fetch_openml(data_id=42165, as_frame=True, return_X_y=True, parser='pandas')
categorical_columns_subset = ['BldgType', 'GarageFinish', 'LotConfig', 'Functional', 'MasVnrType', 'HouseStyle', 'FireplaceQu', 'ExterCond', 'ExterQual', 'PoolQC']
numerical_columns_subset = ['3SsnPorch', 'Fireplaces', 'BsmtHalfBath', 'HalfBath', 'GarageCars', 'TotRmsAbvGrd', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea', 'ScreenPorch']
X = X[categorical_columns_subset + numerical_columns_subset]
X[categorical_columns_subset] = X[categorical_columns_subset].astype('category')
categorical_columns = X.select_dtypes(include='category').columns
n_categorical_features = len(categorical_columns)
n_numerical_features = X.select_dtypes(include='number').shape[1]
print(f'Number of samples: {X.shape[0]}')
print(f'Number of features: {X.shape[1]}')
print(f'Number of categorical features: {n_categorical_features}')
print(f'Number of numerical features: {n_numerical_features}')
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
dropper = make_column_transformer(('drop', make_column_selector(dtype_include='category')), remainder='passthrough')
hist_dropped = make_pipeline(dropper, HistGradientBoostingRegressor(random_state=42))
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = make_column_transformer((OneHotEncoder(sparse_output=False, handle_unknown='ignore'), make_column_selector(dtype_include='category')), remainder='passthrough')
hist_one_hot = make_pipeline(one_hot_encoder, HistGradientBoostingRegressor(random_state=42))
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = make_column_transformer((OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), make_column_selector(dtype_include='category')), remainder='passthrough', verbose_feature_names_out=False)
hist_ordinal = make_pipeline(ordinal_encoder, HistGradientBoostingRegressor(random_state=42))
hist_native = make_pipeline(ordinal_encoder, HistGradientBoostingRegressor(random_state=42, categorical_features=categorical_columns)).set_output(transform='pandas')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
scoring = 'neg_mean_absolute_percentage_error'
n_cv_folds = 3
dropped_result = cross_validate(hist_dropped, X, y, cv=n_cv_folds, scoring=scoring)
one_hot_result = cross_validate(hist_one_hot, X, y, cv=n_cv_folds, scoring=scoring)
ordinal_result = cross_validate(hist_ordinal, X, y, cv=n_cv_folds, scoring=scoring)
native_result = cross_validate(hist_native, X, y, cv=n_cv_folds, scoring=scoring)

def plot_results(figure_title):
    if False:
        while True:
            i = 10
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 8))
    plot_info = [('fit_time', 'Fit times (s)', ax1, None), ('test_score', 'Mean Absolute Percentage Error', ax2, None)]
    (x, width) = (np.arange(4), 0.9)
    for (key, title, ax, y_limit) in plot_info:
        items = [dropped_result[key], one_hot_result[key], ordinal_result[key], native_result[key]]
        mape_cv_mean = [np.mean(np.abs(item)) for item in items]
        mape_cv_std = [np.std(item) for item in items]
        ax.bar(x=x, height=mape_cv_mean, width=width, yerr=mape_cv_std, color=['C0', 'C1', 'C2', 'C3'])
        ax.set(xlabel='Model', title=title, xticks=x, xticklabels=['Dropped', 'One Hot', 'Ordinal', 'Native'], ylim=y_limit)
    fig.suptitle(figure_title)
plot_results('Gradient Boosting on Ames Housing')
for pipe in (hist_dropped, hist_one_hot, hist_ordinal, hist_native):
    pipe.set_params(histgradientboostingregressor__max_depth=3, histgradientboostingregressor__max_iter=15)
dropped_result = cross_validate(hist_dropped, X, y, cv=n_cv_folds, scoring=scoring)
one_hot_result = cross_validate(hist_one_hot, X, y, cv=n_cv_folds, scoring=scoring)
ordinal_result = cross_validate(hist_ordinal, X, y, cv=n_cv_folds, scoring=scoring)
native_result = cross_validate(hist_native, X, y, cv=n_cv_folds, scoring=scoring)
plot_results('Gradient Boosting on Ames Housing (few and small trees)')
plt.show()