"""
=================================
Combine predictors using stacking
=================================

.. currentmodule:: sklearn

Stacking refers to a method to blend estimators. In this strategy, some
estimators are individually fitted on some training data while a final
estimator is trained using the stacked predictions of these base estimators.

In this example, we illustrate the use case in which different regressors are
stacked together and a final linear penalized regressor is used to output the
prediction. We compare the performance of each individual regressor with the
stacking strategy. Stacking slightly improves the overall performance.

"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

def load_ames_housing():
    if False:
        for i in range(10):
            print('nop')
    df = fetch_openml(name='house_prices', as_frame=True, parser='pandas')
    X = df.data
    y = df.target
    features = ['YrSold', 'HeatingQC', 'Street', 'YearRemodAdd', 'Heating', 'MasVnrType', 'BsmtUnfSF', 'Foundation', 'MasVnrArea', 'MSSubClass', 'ExterQual', 'Condition2', 'GarageCars', 'GarageType', 'OverallQual', 'TotalBsmtSF', 'BsmtFinSF1', 'HouseStyle', 'MiscFeature', 'MoSold']
    X = X.loc[:, features]
    (X, y) = shuffle(X, y, random_state=0)
    X = X.iloc[:600]
    y = y.iloc[:600]
    return (X, np.log(y))
(X, y) = load_ames_housing()
from sklearn.compose import make_column_selector
cat_selector = make_column_selector(dtype_include=object)
num_selector = make_column_selector(dtype_include=np.number)
cat_selector(X)
num_selector(X)
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
cat_tree_processor = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-2)
num_tree_processor = SimpleImputer(strategy='mean', add_indicator=True)
tree_preprocessor = make_column_transformer((num_tree_processor, num_selector), (cat_tree_processor, cat_selector))
tree_preprocessor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
cat_linear_processor = OneHotEncoder(handle_unknown='ignore')
num_linear_processor = make_pipeline(StandardScaler(), SimpleImputer(strategy='mean', add_indicator=True))
linear_preprocessor = make_column_transformer((num_linear_processor, num_selector), (cat_linear_processor, cat_selector))
linear_preprocessor
from sklearn.linear_model import LassoCV
lasso_pipeline = make_pipeline(linear_preprocessor, LassoCV())
lasso_pipeline
from sklearn.ensemble import RandomForestRegressor
rf_pipeline = make_pipeline(tree_preprocessor, RandomForestRegressor(random_state=42))
rf_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
gbdt_pipeline = make_pipeline(tree_preprocessor, HistGradientBoostingRegressor(random_state=0))
gbdt_pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
estimators = [('Random Forest', rf_pipeline), ('Lasso', lasso_pipeline), ('Gradient Boosting', gbdt_pipeline)]
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
stacking_regressor
import time
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import cross_val_predict, cross_validate
(fig, axs) = plt.subplots(2, 2, figsize=(9, 7))
axs = np.ravel(axs)
for (ax, (name, est)) in zip(axs, estimators + [('Stacking Regressor', stacking_regressor)]):
    scorers = {'R2': 'r2', 'MAE': 'neg_mean_absolute_error'}
    start_time = time.time()
    scores = cross_validate(est, X, y, scoring=list(scorers.values()), n_jobs=-1, verbose=0)
    elapsed_time = time.time() - start_time
    y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0)
    scores = {key: f"{np.abs(np.mean(scores[f'test_{value}'])):.2f} +- {np.std(scores[f'test_{value}']):.2f}" for (key, value) in scorers.items()}
    display = PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred, kind='actual_vs_predicted', ax=ax, scatter_kwargs={'alpha': 0.2, 'color': 'tab:blue'}, line_kwargs={'color': 'tab:red'})
    ax.set_title(f'{name}\nEvaluation in {elapsed_time:.2f} seconds')
    for (name, score) in scores.items():
        ax.plot([], [], ' ', label=f'{name}: {score}')
    ax.legend(loc='upper left')
plt.suptitle('Single predictors versus stacked predictors')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()