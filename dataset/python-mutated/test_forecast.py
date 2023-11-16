import datetime
import numpy as np
import pandas as pd
from flaml import AutoML
from flaml.automl.task.time_series_task import TimeSeriesTask

def test_forecast_automl(budget=10, estimators_when_no_prophet=['arima', 'sarimax', 'holt-winters']):
    if False:
        while True:
            i = 10
    import statsmodels.api as sm
    data = sm.datasets.co2.load_pandas().data['co2'].resample('MS').mean()
    data = data.bfill().ffill().to_frame().reset_index().rename(columns={'index': 'ds', 'co2': 'y'})
    num_samples = data.shape[0]
    time_horizon = 12
    split_idx = num_samples - time_horizon
    df = data[:split_idx]
    X_test = data[split_idx:]['ds']
    y_test = data[split_idx:]['y']
    automl = AutoML()
    settings = {'time_budget': budget, 'metric': 'mape', 'task': 'ts_forecast', 'log_file_name': 'test/CO2_forecast.log', 'eval_method': 'holdout', 'label': 'y'}
    'The main flaml automl API'
    try:
        import prophet
        automl.fit(dataframe=df, **settings, period=time_horizon)
    except ImportError:
        print('not using prophet due to ImportError')
        automl.fit(dataframe=df, **settings, estimator_list=estimators_when_no_prophet, period=time_horizon)
    ' retrieve best config and best learner'
    print('Best ML leaner:', automl.best_estimator)
    print('Best hyperparmeter config:', automl.best_config)
    print(f'Best mape on validation data: {automl.best_loss}')
    print(f'Training duration of best run: {automl.best_config_train_time}s')
    print(automl.model.estimator)
    ' pickle and save the automl object '
    import pickle
    with open('automl.pkl', 'wb') as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    ' compute predictions of testing dataset '
    y_pred = automl.predict(X_test)
    print('Predicted labels', y_pred)
    print('True labels', y_test)
    ' compute different metric values on testing dataset'
    from flaml.automl.ml import sklearn_metric_loss_score
    mape = sklearn_metric_loss_score('mape', y_pred, y_test)
    print('mape', '=', mape)
    assert mape <= 0.005, 'the mape of flaml should be less than 0.005'
    from flaml.automl.data import get_output_from_log
    (time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history) = get_output_from_log(filename=settings['log_file_name'], time_budget=budget)
    for config in config_history:
        print(config)
    print(automl.resource_attr)
    print(automl.max_resource)
    print(automl.min_resource)
    X_train = df[['ds']]
    y_train = df['y']
    automl = AutoML()
    try:
        automl.fit(X_train=X_train, y_train=y_train, **settings, period=time_horizon)
    except ImportError:
        print('not using prophet due to ImportError')
        automl.fit(X_train=X_train, y_train=y_train, **settings, estimator_list=estimators_when_no_prophet, period=time_horizon)

def test_models(budget=3):
    if False:
        i = 10
        return i + 15
    n = 100
    X = pd.DataFrame({'A': pd.date_range(start='1900-01-01', periods=n, freq='D')})
    y = np.exp(np.random.randn(n))
    task = TimeSeriesTask('ts_forecast')
    for est in task.estimators.keys():
        if est == 'tft':
            continue
        automl = AutoML()
        automl.fit(X_train=X[:72], y_train=y[:72], estimator_list=[est], period=12, task='ts_forecast', time_budget=budget)
        automl.predict(X[72:])

def test_numpy():
    if False:
        print('Hello World!')
    X_train = np.arange('2014-01', '2021-01', dtype='datetime64[M]')
    y_train = np.random.random(size=len(X_train))
    automl = AutoML()
    automl.fit(X_train=X_train[:72], y_train=y_train[:72], period=12, task='ts_forecast', time_budget=3, log_file_name='test/ts_forecast.log', n_splits=3)
    print(automl.predict(X_train[72:]))
    automl = AutoML()
    automl.fit(X_train=X_train[:72], y_train=y_train[:72], period=12, task='ts_forecast', time_budget=1, estimator_list=['arima', 'sarimax'], log_file_name='test/ts_forecast.log')
    print(automl.predict(X_train[72:]))
    print(automl.predict(12))

def test_numpy_large():
    if False:
        while True:
            i = 10
    import numpy as np
    import pandas as pd
    from flaml import AutoML
    X_train = pd.date_range('2017-01-01', periods=70000, freq='T')
    y_train = pd.DataFrame(np.random.randint(6500, 7500, 70000))
    automl = AutoML()
    automl.fit(X_train=X_train[:-10].values, y_train=y_train[:-10].values, period=10, task='ts_forecast', time_budget=10)

def load_multi_dataset():
    if False:
        return 10
    'multivariate time series forecasting dataset'
    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/nyc_energy_consumption.csv')
    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    df = df.set_index('timeStamp')
    df = df.resample('D').mean()
    df['temp'] = df['temp'].fillna(method='ffill')
    df['precip'] = df['precip'].fillna(method='ffill')
    df = df[:-2]
    df = df.reset_index()
    return df

def test_multivariate_forecast_num(budget=5, estimators_when_no_prophet=['arima', 'sarimax', 'holt-winters']):
    if False:
        return 10
    df = load_multi_dataset()
    time_horizon = 180
    num_samples = df.shape[0]
    split_idx = num_samples - time_horizon
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    X_test = test_df[['timeStamp', 'temp', 'precip']]
    y_test = test_df['demand']
    automl = AutoML()
    settings = {'time_budget': budget, 'metric': 'mape', 'task': 'ts_forecast', 'log_file_name': 'test/energy_forecast_numerical.log', 'eval_method': 'holdout', 'log_type': 'all', 'label': 'demand'}
    'The main flaml automl API'
    try:
        import prophet
        automl.fit(dataframe=train_df, **settings, period=time_horizon)
    except ImportError:
        print('not using prophet due to ImportError')
        automl.fit(dataframe=train_df, **settings, estimator_list=estimators_when_no_prophet, period=time_horizon)
    ' retrieve best config and best learner'
    print('Best ML leaner:', automl.best_estimator)
    print('Best hyperparmeter config:', automl.best_config)
    print(f'Best mape on validation data: {automl.best_loss}')
    print(f'Training duration of best run: {automl.best_config_train_time}s')
    print(automl.model.estimator)
    ' pickle and save the automl object '
    import pickle
    with open('automl.pkl', 'wb') as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    ' compute predictions of testing dataset '
    y_pred = automl.predict(X_test)
    print('Predicted labels', y_pred)
    print('True labels', y_test)
    ' compute different metric values on testing dataset'
    from flaml.automl.ml import sklearn_metric_loss_score
    print('mape', '=', sklearn_metric_loss_score('mape', y_pred, y_test))
    from flaml.automl.data import get_output_from_log
    (time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history) = get_output_from_log(filename=settings['log_file_name'], time_budget=budget)
    for config in config_history:
        print(config)
    print(automl.resource_attr)
    print(automl.max_resource)
    print(automl.min_resource)

def load_multi_dataset_cat(time_horizon):
    if False:
        print('Hello World!')
    df = load_multi_dataset()
    df = df[['timeStamp', 'demand', 'temp']]

    def season(date):
        if False:
            print('Hello World!')
        date = (date.month, date.day)
        spring = (3, 20)
        summer = (6, 21)
        fall = (9, 22)
        winter = (12, 21)
        if date < spring or date >= winter:
            return 'winter'
        elif spring <= date < summer:
            return 'spring'
        elif summer <= date < fall:
            return 'summer'
        elif fall <= date < winter:
            return 'fall'

    def get_monthly_avg(data):
        if False:
            while True:
                i = 10
        data['month'] = data['timeStamp'].dt.month
        data = data[['month', 'temp']].groupby('month')
        data = data.agg({'temp': 'mean'})
        return data
    monthly_avg = get_monthly_avg(df).to_dict().get('temp')

    def above_monthly_avg(date, temp):
        if False:
            i = 10
            return i + 15
        month = date.month
        if temp > monthly_avg.get(month):
            return 1
        else:
            return 0
    df['season'] = df['timeStamp'].apply(season)
    df['above_monthly_avg'] = df.apply(lambda x: above_monthly_avg(x['timeStamp'], x['temp']), axis=1)
    num_samples = df.shape[0]
    split_idx = num_samples - time_horizon
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    del train_df['temp'], train_df['month']
    return (train_df, test_df)

def test_multivariate_forecast_cat(budget=5, estimators_when_no_prophet=['arima', 'sarimax', 'holt-winters']):
    if False:
        i = 10
        return i + 15
    time_horizon = 180
    (train_df, test_df) = load_multi_dataset_cat(time_horizon)
    X_test = test_df[['timeStamp', 'season', 'above_monthly_avg']]
    y_test = test_df['demand']
    automl = AutoML()
    settings = {'time_budget': budget, 'metric': 'mape', 'task': 'ts_forecast', 'log_file_name': 'test/energy_forecast_categorical.log', 'eval_method': 'holdout', 'log_type': 'all', 'label': 'demand'}
    'The main flaml automl API'
    try:
        import prophet
        automl.fit(dataframe=train_df, **settings, period=time_horizon)
    except ImportError:
        print('not using prophet due to ImportError')
        automl.fit(dataframe=train_df, **settings, estimator_list=estimators_when_no_prophet, period=time_horizon)
    ' retrieve best config and best learner'
    print('Best ML leaner:', automl.best_estimator)
    print('Best hyperparmeter config:', automl.best_config)
    print(f'Best mape on validation data: {automl.best_loss}')
    print(f'Training duration of best run: {automl.best_config_train_time}s')
    print(automl.model.estimator)
    ' pickle and save the automl object '
    import pickle
    with open('automl.pkl', 'wb') as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    ' compute predictions of testing dataset '
    y_pred = automl.predict(X_test)
    print('Predicted labels', y_pred)
    print('True labels', y_test)
    ' compute different metric values on testing dataset'
    from flaml.automl.ml import sklearn_metric_loss_score
    print('mape', '=', sklearn_metric_loss_score('mape', y_pred, y_test))
    print('rmse', '=', sklearn_metric_loss_score('rmse', y_pred, y_test))
    print('mse', '=', sklearn_metric_loss_score('mse', y_pred, y_test))
    print('mae', '=', sklearn_metric_loss_score('mae', y_pred, y_test))
    from flaml.automl.data import get_output_from_log
    (time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history) = get_output_from_log(filename=settings['log_file_name'], time_budget=budget)
    for config in config_history:
        print(config)
    print(automl.resource_attr)
    print(automl.max_resource)
    print(automl.min_resource)

def test_forecast_classification(budget=5):
    if False:
        print('Hello World!')
    from hcrystalball.utils import get_sales_data
    time_horizon = 30
    df = get_sales_data(n_dates=180, n_assortments=1, n_states=1, n_stores=1)
    df = df[['Sales', 'Open', 'Promo', 'Promo2']]
    import numpy as np
    df['above_mean_sales'] = np.where(df['Sales'] > df['Sales'].mean(), 1, 0)
    df.reset_index(inplace=True)
    train_df = df[:-time_horizon]
    test_df = df[-time_horizon:]
    (X_train, X_test) = (train_df[['Date', 'Open', 'Promo', 'Promo2']], test_df[['Date', 'Open', 'Promo', 'Promo2']])
    (y_train, y_test) = (train_df['above_mean_sales'], test_df['above_mean_sales'])
    automl = AutoML()
    settings = {'time_budget': budget, 'metric': 'accuracy', 'task': 'ts_forecast_classification', 'log_file_name': 'test/sales_classification_forecast.log', 'eval_method': 'holdout'}
    'The main flaml automl API'
    automl.fit(X_train=X_train, y_train=y_train, **settings, period=time_horizon)
    ' retrieve best config and best learner'
    print('Best ML leaner:', automl.best_estimator)
    print('Best hyperparmeter config:', automl.best_config)
    print(f'Best mape on validation data: {automl.best_loss}')
    print(f'Training duration of best run: {automl.best_config_train_time}s')
    print(automl.model.estimator)
    ' pickle and save the automl object '
    import pickle
    with open('automl.pkl', 'wb') as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    ' compute predictions of testing dataset '
    y_pred = automl.predict(X_test)
    ' compute different metric values on testing dataset'
    from flaml.automl.ml import sklearn_metric_loss_score
    print(y_test)
    print(y_pred)
    print('accuracy', '=', 1 - sklearn_metric_loss_score('accuracy', y_pred, y_test))
    from flaml.automl.data import get_output_from_log
    (time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history) = get_output_from_log(filename=settings['log_file_name'], time_budget=budget)
    for config in config_history:
        print(config)
    print(automl.resource_attr)
    print(automl.max_resource)
    print(automl.min_resource)

def get_stalliion_data():
    if False:
        while True:
            i = 10
    from pytorch_forecasting.data.examples import get_stallion_data
    data = get_stallion_data()
    data['time_idx'] = data['date'].dt.year * 12 + data['date'].dt.month
    data['time_idx'] -= data['time_idx'].min()
    data['month'] = data.date.dt.month.astype(str).astype('category')
    data['log_volume'] = np.log(data.volume + 1e-08)
    data['avg_volume_by_sku'] = data.groupby(['time_idx', 'sku'], observed=True).volume.transform('mean')
    data['avg_volume_by_agency'] = data.groupby(['time_idx', 'agency'], observed=True).volume.transform('mean')
    special_days = ['easter_day', 'good_friday', 'new_year', 'christmas', 'labor_day', 'independence_day', 'revolution_day_memorial', 'regional_games', 'beer_capital', 'music_fest']
    data[special_days] = data[special_days].apply(lambda x: x.map({0: '-', 1: x.name})).astype('category')
    return (data, special_days)

def test_forecast_panel(budget=5):
    if False:
        print('Hello World!')
    (data, special_days) = get_stalliion_data()
    time_horizon = 6
    training_cutoff = data['time_idx'].max() - time_horizon
    data['time_idx'] = data['time_idx'].astype('int')
    ts_col = data.pop('date')
    data.insert(0, 'date', ts_col)
    data = data.sort_values(['agency', 'sku', 'date'])
    X_train = data[lambda x: x.time_idx <= training_cutoff]
    X_test = data[lambda x: x.time_idx > training_cutoff]
    y_train = X_train.pop('volume')
    y_test = X_test.pop('volume')
    automl = AutoML()
    settings = {'time_budget': budget, 'metric': 'mape', 'task': 'ts_forecast_panel', 'log_file_name': 'test/stallion_forecast.log', 'eval_method': 'holdout'}
    fit_kwargs_by_estimator = {'tft': {'max_encoder_length': 24, 'static_categoricals': ['agency', 'sku'], 'static_reals': ['avg_population_2017', 'avg_yearly_household_income_2017'], 'time_varying_known_categoricals': ['special_days', 'month'], 'variable_groups': {'special_days': special_days}, 'time_varying_known_reals': ['time_idx', 'price_regular', 'discount_in_percent'], 'time_varying_unknown_categoricals': [], 'time_varying_unknown_reals': ['volume', 'log_volume', 'industry_volume', 'soda_volume', 'avg_max_temp', 'avg_volume_by_agency', 'avg_volume_by_sku'], 'batch_size': 256, 'max_epochs': 1, 'gpu_per_trial': -1}}
    'The main flaml automl API'
    automl.fit(X_train=X_train, y_train=y_train, **settings, period=time_horizon, group_ids=['agency', 'sku'], fit_kwargs_by_estimator=fit_kwargs_by_estimator)
    ' retrieve best config and best learner'
    print('Best ML leaner:', automl.best_estimator)
    print('Best hyperparmeter config:', automl.best_config)
    print(f'Best mape on validation data: {automl.best_loss}')
    print(f'Training duration of best run: {automl.best_config_train_time}s')
    print(automl.model.estimator)
    ' pickle and save the automl object '
    import pickle
    with open('automl.pkl', 'wb') as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    ' compute predictions of testing dataset '
    y_pred = automl.predict(X_test)
    ' compute different metric values on testing dataset'
    from flaml.automl.ml import sklearn_metric_loss_score
    print(y_test)
    print(y_pred)
    print('mape', '=', sklearn_metric_loss_score('mape', y_pred, y_test))

    def smape(y_pred, y_test):
        if False:
            while True:
                i = 10
        import numpy as np
        (y_test, y_pred) = (np.array(y_test), np.array(y_pred))
        return round(np.mean(np.abs(y_pred - y_test) / ((np.abs(y_pred) + np.abs(y_test)) / 2)) * 100, 2)
    print('smape', '=', smape(y_pred, y_test))
    from flaml.automl.data import get_output_from_log
    (time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history) = get_output_from_log(filename=settings['log_file_name'], time_budget=budget)
    for config in config_history:
        print(config)
    print(automl.resource_attr)
    print(automl.max_resource)
    print(automl.min_resource)

def test_cv_step():
    if False:
        return 10
    n = 300
    time_col = 'date'
    df = pd.DataFrame({time_col: pd.date_range(start='1/1/2001', periods=n, freq='D'), 'y': np.sin(np.linspace(start=0, stop=200, num=n))})

    def split_by_date(df: pd.DataFrame, dt: datetime.date):
        if False:
            print('Hello World!')
        dt = datetime.datetime(dt.year, dt.month, dt.day)
        return (df[df[time_col] <= dt], df[df[time_col] > dt])
    horizon = 60
    data_end = df.date.max()
    train_end = data_end - datetime.timedelta(days=horizon)
    (train_df, val_df) = split_by_date(df, train_end)
    from flaml import AutoML
    tgts = ['y']
    preds = {}
    for tgt in tgts:
        features = []
        automl = AutoML(time_budget=5, metric='mae', task='ts_forecast', eval_method='cv')
        automl.fit(dataframe=train_df[[time_col] + features + [tgt]], label=tgt, period=horizon, time_col=time_col, verbose=4, n_splits=5, cv_step_size=5)
        pred = automl.predict(val_df)
        if isinstance(pred, pd.DataFrame):
            pred = pred[tgt]
        assert not np.isnan(pred.sum())
        import matplotlib.pyplot as plt
        preds[tgt] = pred
    print('yahoo!')
if __name__ == '__main__':
    test_forecast_panel(5)