from bigdl.chronos.forecaster.abstract import Forecaster
from bigdl.chronos.model.prophet import ProphetModel

class ProphetForecaster(Forecaster):
    """
    Example:
        >>> #The dataset is split into data, validation_data
        >>> model = ProphetForecaster(changepoint_prior_scale=0.05, seasonality_mode='additive')
        >>> model.fit(data, validation_data)
        >>> predict_result = model.predict(horizon=24)
    """

    def __init__(self, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, holidays_prior_scale=10.0, seasonality_mode='additive', changepoint_range=0.8, metric='mse'):
        if False:
            print('Hello World!')
        '\n        Build a Prophet Forecast Model.\n        User can customize changepoint_prior_scale, seasonality_prior_scale,\n        holidays_prior_scale, seasonality_mode, changepoint_range and metric\n        of the Prophet model, for details of the Prophet model hyperparameters, refer to\n        https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning.\n\n        :param changepoint_prior_scale: hyperparameter changepoint_prior_scale for the\n            Prophet model.\n        :param seasonality_prior_scale: hyperparameter seasonality_prior_scale for the\n            Prophet model.\n        :param holidays_prior_scale: hyperparameter holidays_prior_scale for the\n            Prophet model.\n        :param seasonality_mode: hyperparameter seasonality_mode for the\n            Prophet model.\n        :param changepoint_range: hyperparameter changepoint_range for the\n            Prophet model.\n        :param metric: the metric for validation and evaluation. For regression, we support\n            Mean Squared Error: ("mean_squared_error", "MSE" or "mse"),\n            Mean Absolute Error: ("mean_absolute_error","MAE" or "mae"),\n            Mean Absolute Percentage Error: ("mean_absolute_percentage_error", "MAPE", "mape")\n            Cosine Proximity: ("cosine_proximity", "cosine")\n        '
        self.model_config = {'changepoint_prior_scale': changepoint_prior_scale, 'seasonality_prior_scale': seasonality_prior_scale, 'holidays_prior_scale': holidays_prior_scale, 'seasonality_mode': seasonality_mode, 'changepoint_range': changepoint_range, 'metric': metric}
        self.internal = ProphetModel()
        super().__init__()

    def fit(self, data, validation_data=None):
        if False:
            print('Hello World!')
        "\n        Fit(Train) the forecaster.\n\n        :param data: training data, a pandas dataframe with Td rows,\n            and 2 columns, with column 'ds' indicating date and column 'y' indicating value\n            and Td is the time dimension\n        :param validation_data: evaluation data, should be the same type as data\n\n        :return: the evaluation metric value\n        "
        self._check_data(data, validation_data)
        return self.internal.fit_eval(data=data, validation_data=validation_data, **self.model_config)

    def _check_data(self, data, validation_data):
        if False:
            for i in range(10):
                print('nop')
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError('ds' in data.columns and 'y' in data.columns, "data should be a pandas dataframe that has at least 2 columns 'ds' and 'y'.")
        if validation_data is not None:
            invalidInputError('ds' in validation_data.columns and 'y' in validation_data.columns, "validation_data should be a dataframe that has at least 2 columns 'ds' and 'y'.")

    def predict(self, horizon=1, freq='D', ds_data=None):
        if False:
            print('Hello World!')
        '\n        Predict using a trained forecaster.\n\n        :param horizon: the number of steps forward to predict, the value defaults to 1.\n        :param freq: the freqency of the predicted dataframe, defaulted to day("D"),\n               the frequency can be anything from the pandas list of frequency strings here:\n               https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases\n        :param ds_data: a dataframe that has 1 column \'ds\' indicating date.\n\n        :return: A pandas DataFrame of length horizon,\n                 including "trend" and "seasonality" and inference values, etc.\n                 where the "yhat" column is the inference value.\n        '
        if self.internal.model is None:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False, 'You must call fit or restore first before calling predict!')
        return self.internal.predict(horizon=horizon, freq=freq, ds_data=ds_data)

    def evaluate(self, data, metrics=['mse']):
        if False:
            while True:
                i = 10
        "\n        Evaluate using a trained forecaster.\n\n        :param data: evaluation data, a pandas dataframe with Td rows,\n            and 2 columns, with column 'ds' indicating date and column 'y' indicating value\n            and Td is the time dimension\n        :param metrics: A list contains metrics for test/valid data.\n\n        :return: A list of evaluation results. Calculation results for each metrics.\n        "
        from bigdl.nano.utils.common import invalidInputError
        if data is None:
            invalidInputError(False, 'Input invalid data of None')
        if self.internal.model is None:
            invalidInputError(False, 'You must call fit or restore first before calling evaluate!')
        return self.internal.evaluate(target=data, metrics=metrics)

    def save(self, checkpoint_file):
        if False:
            return 10
        '\n        Save the forecaster.\n\n        :param checkpoint_file: The location you want to save the forecaster, should be a json file\n        '
        if self.internal.model is None:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False, 'You must call fit or restore first before calling save!')
        self.internal.save(checkpoint_file)

    def restore(self, checkpoint_file):
        if False:
            print('Hello World!')
        '\n        Restore the forecaster.\n\n        :param checkpoint_file: The checkpoint file location you want to load the forecaster.\n        '
        self.internal.restore(checkpoint_file)