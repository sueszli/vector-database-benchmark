from bigdl.chronos.forecaster.abstract import Forecaster
from bigdl.chronos.model.arima import ARIMAModel

class ARIMAForecaster(Forecaster):
    """
    Example:
        >>> #The dataset is split into data, validation_data
        >>> model = ARIMAForecaster(p=2, q=2, seasonality_mode=False)
        >>> model.fit(data, validation_data)
        >>> predict_result = model.predict(horizon=24)
    """

    def __init__(self, p=2, q=2, seasonality_mode=True, P=3, Q=1, m=7, metric='mse'):
        if False:
            while True:
                i = 10
        '\n        Build a ARIMA Forecast Model.\n        User can customize p, q, seasonality_mode, P, Q, m, metric for the ARIMA model,\n        the differencing term (d) and seasonal differencing term (D) are automatically\n        estimated from the data. For details of the ARIMA model hyperparameters, refer to\n        https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA.\n\n        :param p: hyperparameter p for the ARIMA model.\n        :param q: hyperparameter q for the ARIMA model.\n        :param seasonality_mode: hyperparameter seasonality_mode for the ARIMA model.\n        :param P: hyperparameter P for the ARIMA model.\n        :param Q: hyperparameter Q for the ARIMA model.\n        :param m: hyperparameter m for the ARIMA model.\n        :param metric: the metric for validation and evaluation. For regression, we support\n            Mean Squared Error: ("mean_squared_error", "MSE" or "mse"),\n            Mean Absolute Error: ("mean_absolute_error","MAE" or "mae"),\n            Mean Absolute Percentage Error: ("mean_absolute_percentage_error", "MAPE", "mape")\n            Cosine Proximity: ("cosine_proximity", "cosine")\n        '
        self.model_config = {'p': p, 'q': q, 'seasonality_mode': seasonality_mode, 'P': P, 'Q': Q, 'm': m, 'metric': metric}
        self.internal = ARIMAModel()
        super().__init__()

    def fit(self, data, validation_data):
        if False:
            print('Hello World!')
        '\n        Fit(Train) the forecaster.\n\n        :param data: A 1-D numpy array as the training data\n        :param validation_data: A 1-D numpy array as the evaluation data\n        '
        self._check_data(data, validation_data)
        data = data.reshape(-1, 1)
        return self.internal.fit_eval(data=data, validation_data=validation_data, **self.model_config)

    def _check_data(self, data, validation_data):
        if False:
            i = 10
            return i + 15
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError(data.ndim == 1, 'data should be an 1-D array),Got data dimension of {}.'.format(data.ndim))
        invalidInputError(validation_data.ndim == 1, 'validation_data should be an 1-D array),Got validation_data dimension of {}.'.format(validation_data.ndim))

    def predict(self, horizon, rolling=False):
        if False:
            while True:
                i = 10
        '\n        Predict using a trained forecaster.\n\n        :param horizon: the number of steps forward to predict\n        :param rolling: whether to use rolling prediction\n\n        :return: A list in length of horizon reflects the predict result.\n        '
        if self.internal.model is None:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False, 'You must call fit or restore first before calling predict!')
        return self.internal.predict(horizon=horizon, rolling=rolling)

    def evaluate(self, validation_data, metrics=['mse'], rolling=False):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate using a trained forecaster.\n\n        :param validation_data: A 1-D numpy array as the evaluation data\n        :param metrics: A list contains metrics for test/valid data.\n\n        :return: A list in length of len(metrics), where states the metrics in order.\n        '
        from bigdl.nano.utils.common import invalidInputError
        if validation_data is None:
            invalidInputError(False, 'Input invalid validation_data of None')
        if self.internal.model is None:
            invalidInputError(False, 'You must call fit or restore first before calling evaluate!')
        return self.internal.evaluate(validation_data, metrics=metrics, rolling=rolling)

    def save(self, checkpoint_file):
        if False:
            i = 10
            return i + 15
        '\n        Save the forecaster.\n\n        :param checkpoint_file: The location you want to save the forecaster.\n        '
        if self.internal.model is None:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False, 'You must call fit or restore first before calling save!')
        self.internal.save(checkpoint_file)

    def restore(self, checkpoint_file):
        if False:
            print('Hello World!')
        '\n        Restore the forecaster.\n\n        :param checkpoint_file: The checkpoint file location you want to load the forecaster.\n        '
        self.internal.restore(checkpoint_file)