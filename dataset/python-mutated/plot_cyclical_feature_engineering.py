"""
================================
Time-related feature engineering
================================

This notebook introduces different strategies to leverage time-related features
for a bike sharing demand regression task that is highly dependent on business
cycles (days, weeks, months) and yearly season cycles.

In the process, we introduce how to perform periodic feature engineering using
the :class:`sklearn.preprocessing.SplineTransformer` class and its
`extrapolation="periodic"` option.

"""
from sklearn.datasets import fetch_openml
bike_sharing = fetch_openml('Bike_Sharing_Demand', version=2, as_frame=True, parser='pandas')
df = bike_sharing.frame
import matplotlib.pyplot as plt
(fig, ax) = plt.subplots(figsize=(12, 4))
average_week_demand = df.groupby(['weekday', 'hour'])['count'].mean()
average_week_demand.plot(ax=ax)
_ = ax.set(title='Average hourly bike demand during the week', xticks=[i * 24 for i in range(7)], xticklabels=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'], xlabel='Time of the week', ylabel='Number of bike rentals')
df['count'].max()
y = df['count'] / df['count'].max()
(fig, ax) = plt.subplots(figsize=(12, 4))
y.hist(bins=30, ax=ax)
_ = ax.set(xlabel='Fraction of rented fleet demand', ylabel='Number of hours')
X = df.drop('count', axis='columns')
X
X['weather'].value_counts()
X['weather'].replace(to_replace='heavy_rain', value='rain', inplace=True)
X['weather'].value_counts()
X['season'].value_counts()
from sklearn.model_selection import TimeSeriesSplit
ts_cv = TimeSeriesSplit(n_splits=5, gap=48, max_train_size=10000, test_size=1000)
all_splits = list(ts_cv.split(X, y))
(train_0, test_0) = all_splits[0]
X.iloc[test_0]
X.iloc[train_0]
(train_4, test_4) = all_splits[4]
X.iloc[test_4]
X.iloc[train_4]
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
categorical_columns = ['weather', 'season', 'holiday', 'workingday']
categories = [['clear', 'misty', 'rain'], ['spring', 'summer', 'fall', 'winter'], ['False', 'True'], ['False', 'True']]
ordinal_encoder = OrdinalEncoder(categories=categories)
gbrt_pipeline = make_pipeline(ColumnTransformer(transformers=[('categorical', ordinal_encoder, categorical_columns)], remainder='passthrough', verbose_feature_names_out=False), HistGradientBoostingRegressor(max_iter=300, early_stopping=True, validation_fraction=0.1, categorical_features=categorical_columns, random_state=42)).set_output(transform='pandas')
import numpy as np

def evaluate(model, X, y, cv, model_prop=None, model_step=None):
    if False:
        i = 10
        return i + 15
    cv_results = cross_validate(model, X, y, cv=cv, scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error'], return_estimator=model_prop is not None)
    if model_prop is not None:
        if model_step is not None:
            values = [getattr(m[model_step], model_prop) for m in cv_results['estimator']]
        else:
            values = [getattr(m, model_prop) for m in cv_results['estimator']]
        print(f'Mean model.{model_prop} = {np.mean(values)}')
    mae = -cv_results['test_neg_mean_absolute_error']
    rmse = -cv_results['test_neg_root_mean_squared_error']
    print(f'Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\nRoot Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}')
evaluate(gbrt_pipeline, X, y, cv=ts_cv, model_prop='n_iter_', model_step='histgradientboostingregressor')
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
alphas = np.logspace(-6, 6, 25)
naive_linear_pipeline = make_pipeline(ColumnTransformer(transformers=[('categorical', one_hot_encoder, categorical_columns)], remainder=MinMaxScaler()), RidgeCV(alphas=alphas))
evaluate(naive_linear_pipeline, X, y, cv=ts_cv, model_prop='alpha_', model_step='ridgecv')
one_hot_linear_pipeline = make_pipeline(ColumnTransformer(transformers=[('categorical', one_hot_encoder, categorical_columns), ('one_hot_time', one_hot_encoder, ['hour', 'weekday', 'month'])], remainder=MinMaxScaler()), RidgeCV(alphas=alphas))
evaluate(one_hot_linear_pipeline, X, y, cv=ts_cv)
from sklearn.preprocessing import FunctionTransformer

def sin_transformer(period):
    if False:
        i = 10
        return i + 15
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    if False:
        while True:
            i = 10
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
import pandas as pd
hour_df = pd.DataFrame(np.arange(26).reshape(-1, 1), columns=['hour'])
hour_df['hour_sin'] = sin_transformer(24).fit_transform(hour_df)['hour']
hour_df['hour_cos'] = cos_transformer(24).fit_transform(hour_df)['hour']
hour_df.plot(x='hour')
_ = plt.title("Trigonometric encoding for the 'hour' feature")
(fig, ax) = plt.subplots(figsize=(7, 5))
sp = ax.scatter(hour_df['hour_sin'], hour_df['hour_cos'], c=hour_df['hour'])
ax.set(xlabel='sin(hour)', ylabel='cos(hour)')
_ = fig.colorbar(sp)
cyclic_cossin_transformer = ColumnTransformer(transformers=[('categorical', one_hot_encoder, categorical_columns), ('month_sin', sin_transformer(12), ['month']), ('month_cos', cos_transformer(12), ['month']), ('weekday_sin', sin_transformer(7), ['weekday']), ('weekday_cos', cos_transformer(7), ['weekday']), ('hour_sin', sin_transformer(24), ['hour']), ('hour_cos', cos_transformer(24), ['hour'])], remainder=MinMaxScaler())
cyclic_cossin_linear_pipeline = make_pipeline(cyclic_cossin_transformer, RidgeCV(alphas=alphas))
evaluate(cyclic_cossin_linear_pipeline, X, y, cv=ts_cv)
from sklearn.preprocessing import SplineTransformer

def periodic_spline_transformer(period, n_splines=None, degree=3):
    if False:
        print('Hello World!')
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1
    return SplineTransformer(degree=degree, n_knots=n_knots, knots=np.linspace(0, period, n_knots).reshape(n_knots, 1), extrapolation='periodic', include_bias=True)
hour_df = pd.DataFrame(np.linspace(0, 26, 1000).reshape(-1, 1), columns=['hour'])
splines = periodic_spline_transformer(24, n_splines=12).fit_transform(hour_df)
splines_df = pd.DataFrame(splines, columns=[f'spline_{i}' for i in range(splines.shape[1])])
pd.concat([hour_df, splines_df], axis='columns').plot(x='hour', cmap=plt.cm.tab20b)
_ = plt.title("Periodic spline-based encoding for the 'hour' feature")
cyclic_spline_transformer = ColumnTransformer(transformers=[('categorical', one_hot_encoder, categorical_columns), ('cyclic_month', periodic_spline_transformer(12, n_splines=6), ['month']), ('cyclic_weekday', periodic_spline_transformer(7, n_splines=3), ['weekday']), ('cyclic_hour', periodic_spline_transformer(24, n_splines=12), ['hour'])], remainder=MinMaxScaler())
cyclic_spline_linear_pipeline = make_pipeline(cyclic_spline_transformer, RidgeCV(alphas=alphas))
evaluate(cyclic_spline_linear_pipeline, X, y, cv=ts_cv)
naive_linear_pipeline.fit(X.iloc[train_0], y.iloc[train_0])
naive_linear_predictions = naive_linear_pipeline.predict(X.iloc[test_0])
one_hot_linear_pipeline.fit(X.iloc[train_0], y.iloc[train_0])
one_hot_linear_predictions = one_hot_linear_pipeline.predict(X.iloc[test_0])
cyclic_cossin_linear_pipeline.fit(X.iloc[train_0], y.iloc[train_0])
cyclic_cossin_linear_predictions = cyclic_cossin_linear_pipeline.predict(X.iloc[test_0])
cyclic_spline_linear_pipeline.fit(X.iloc[train_0], y.iloc[train_0])
cyclic_spline_linear_predictions = cyclic_spline_linear_pipeline.predict(X.iloc[test_0])
last_hours = slice(-96, None)
(fig, ax) = plt.subplots(figsize=(12, 4))
fig.suptitle('Predictions by linear models')
ax.plot(y.iloc[test_0].values[last_hours], 'x-', alpha=0.2, label='Actual demand', color='black')
ax.plot(naive_linear_predictions[last_hours], 'x-', label='Ordinal time features')
ax.plot(cyclic_cossin_linear_predictions[last_hours], 'x-', label='Trigonometric time features')
ax.plot(cyclic_spline_linear_predictions[last_hours], 'x-', label='Spline-based time features')
ax.plot(one_hot_linear_predictions[last_hours], 'x-', label='One-hot time features')
_ = ax.legend()
naive_linear_pipeline[:-1].transform(X).shape
one_hot_linear_pipeline[:-1].transform(X).shape
cyclic_cossin_linear_pipeline[:-1].transform(X).shape
cyclic_spline_linear_pipeline[:-1].transform(X).shape
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures
hour_workday_interaction = make_pipeline(ColumnTransformer([('cyclic_hour', periodic_spline_transformer(24, n_splines=8), ['hour']), ('workingday', FunctionTransformer(lambda x: x == 'True'), ['workingday'])]), PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
cyclic_spline_interactions_pipeline = make_pipeline(FeatureUnion([('marginal', cyclic_spline_transformer), ('interactions', hour_workday_interaction)]), RidgeCV(alphas=alphas))
evaluate(cyclic_spline_interactions_pipeline, X, y, cv=ts_cv)
from sklearn.kernel_approximation import Nystroem
cyclic_spline_poly_pipeline = make_pipeline(cyclic_spline_transformer, Nystroem(kernel='poly', degree=2, n_components=300, random_state=0), RidgeCV(alphas=alphas))
evaluate(cyclic_spline_poly_pipeline, X, y, cv=ts_cv)
one_hot_poly_pipeline = make_pipeline(ColumnTransformer(transformers=[('categorical', one_hot_encoder, categorical_columns), ('one_hot_time', one_hot_encoder, ['hour', 'weekday', 'month'])], remainder='passthrough'), Nystroem(kernel='poly', degree=2, n_components=300, random_state=0), RidgeCV(alphas=alphas))
evaluate(one_hot_poly_pipeline, X, y, cv=ts_cv)
gbrt_pipeline.fit(X.iloc[train_0], y.iloc[train_0])
gbrt_predictions = gbrt_pipeline.predict(X.iloc[test_0])
one_hot_poly_pipeline.fit(X.iloc[train_0], y.iloc[train_0])
one_hot_poly_predictions = one_hot_poly_pipeline.predict(X.iloc[test_0])
cyclic_spline_poly_pipeline.fit(X.iloc[train_0], y.iloc[train_0])
cyclic_spline_poly_predictions = cyclic_spline_poly_pipeline.predict(X.iloc[test_0])
last_hours = slice(-96, None)
(fig, ax) = plt.subplots(figsize=(12, 4))
fig.suptitle('Predictions by non-linear regression models')
ax.plot(y.iloc[test_0].values[last_hours], 'x-', alpha=0.2, label='Actual demand', color='black')
ax.plot(gbrt_predictions[last_hours], 'x-', label='Gradient Boosted Trees')
ax.plot(one_hot_poly_predictions[last_hours], 'x-', label='One-hot + polynomial kernel')
ax.plot(cyclic_spline_poly_predictions[last_hours], 'x-', label='Splines + polynomial kernel')
_ = ax.legend()
from sklearn.metrics import PredictionErrorDisplay
(fig, axes) = plt.subplots(nrows=2, ncols=3, figsize=(13, 7), sharex=True, sharey='row')
fig.suptitle('Non-linear regression models', y=1.0)
predictions = [one_hot_poly_predictions, cyclic_spline_poly_predictions, gbrt_predictions]
labels = ['One hot +\npolynomial kernel', 'Splines +\npolynomial kernel', 'Gradient Boosted\nTrees']
plot_kinds = ['actual_vs_predicted', 'residual_vs_predicted']
for (axis_idx, kind) in enumerate(plot_kinds):
    for (ax, pred, label) in zip(axes[axis_idx], predictions, labels):
        disp = PredictionErrorDisplay.from_predictions(y_true=y.iloc[test_0], y_pred=pred, kind=kind, scatter_kwargs={'alpha': 0.3}, ax=ax)
        ax.set_xticks(np.linspace(0, 1, num=5))
        if axis_idx == 0:
            ax.set_yticks(np.linspace(0, 1, num=5))
            ax.legend(['Best model', label], loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)
        ax.set_aspect('equal', adjustable='box')
plt.show()