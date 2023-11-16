import pandas as pd
import warnings
from bigdl.chronos.model.prophet import ProphetBuilder, ProphetModel
from bigdl.chronos.autots.utils import recalculate_n_sampling

class AutoProphet:

    def __init__(self, changepoint_prior_scale=None, seasonality_prior_scale=None, holidays_prior_scale=None, seasonality_mode=None, changepoint_range=None, metric='mse', metric_mode=None, logs_dir='/tmp/auto_prophet_logs', cpus_per_trial=1, name='auto_prophet', remote_dir=None, load_dir=None, **prophet_config):
        if False:
            print('Hello World!')
        '\n        Create an automated Prophet Model.\n        User need to specify either the exact value or the search space of the\n        Prophet model hyperparameters. For details of the Prophet model hyperparameters, refer to\n        https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning.\n\n        :param changepoint_prior_scale: Int or hp sampling function from an integer space\n            for hyperparameter changepoint_prior_scale for the Prophet model.\n            For hp sampling, see bigdl.chronos.orca.automl.hp for more details.\n            e.g. hp.loguniform(0.001, 0.5).\n        :param seasonality_prior_scale: hyperparameter seasonality_prior_scale for the\n            Prophet model.\n            e.g. hp.loguniform(0.01, 10).\n        :param holidays_prior_scale: hyperparameter holidays_prior_scale for the\n            Prophet model.\n            e.g. hp.loguniform(0.01, 10).\n        :param seasonality_mode: hyperparameter seasonality_mode for the\n            Prophet model.\n            e.g. hp.choice([\'additive\', \'multiplicative\']).\n        :param changepoint_range: hyperparameter changepoint_range for the\n            Prophet model.\n            e.g. hp.uniform(0.8, 0.95).\n        :param metric: String or customized evaluation metric function.\n            If string, metric is the evaluation metric name to optimize, e.g. "mse".\n            If callable function, it signature should be func(y_true, y_pred), where y_true and\n            y_pred are numpy ndarray. The function should return a float value as evaluation result.\n        :param metric_mode: One of ["min", "max"]. "max" means greater metric value is better.\n            You have to specify metric_mode if you use a customized metric function.\n            You don\'t have to specify metric_mode if you use the built-in metric in\n            bigdl.orca.automl.metrics.Evaluator.\n        :param logs_dir: Local directory to save logs and results. It defaults to\n            "/tmp/auto_prophet_logs"\n        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.\n        :param name: name of the AutoProphet. It defaults to "auto_prophet"\n        :param remote_dir: String. Remote directory to sync training results and checkpoints. It\n            defaults to None and doesn\'t take effects while running in local. While running in\n            cluster, it defaults to "hdfs:///tmp/{name}".\n        :param load_dir: Load the ckpt from load_dir. The value defaults to None.\n\n        :param prophet_config: Other Prophet hyperparameters.\n        '
        if load_dir:
            self.best_model = ProphetModel()
            self.best_model.restore(load_dir)
        try:
            from bigdl.orca.automl.auto_estimator import AutoEstimator
            import bigdl.orca.automl.hp as hp
            self.search_space = {'changepoint_prior_scale': hp.grid_search([0.005, 0.05, 0.1, 0.5]) if changepoint_prior_scale is None else changepoint_prior_scale, 'seasonality_prior_scale': hp.grid_search([0.01, 0.1, 1.0, 10.0]) if seasonality_prior_scale is None else seasonality_prior_scale, 'holidays_prior_scale': hp.loguniform(0.01, 10) if holidays_prior_scale is None else holidays_prior_scale, 'seasonality_mode': hp.choice(['additive', 'multiplicative']) if seasonality_mode is None else seasonality_mode, 'changepoint_range': hp.uniform(0.8, 0.95) if changepoint_range is None else changepoint_range}
            self.search_space.update(prophet_config)
            self.metric = metric
            self.metric_mode = metric_mode
            model_builder = ProphetBuilder()
            self.auto_est = AutoEstimator(model_builder=model_builder, logs_dir=logs_dir, resources_per_trial={'cpu': cpus_per_trial}, remote_dir=remote_dir, name=name)
        except ImportError:
            warnings.warn('You need to install `bigdl-orca[automl]` to use `fit` function.')

    def fit(self, data, cross_validation=True, expect_horizon=None, freq=None, metric_threshold=None, n_sampling=16, search_alg=None, search_alg_params=None, scheduler=None, scheduler_params=None):
        if False:
            print('Hello World!')
        '\n        Automatically fit the model and search for the best hyperparameters.\n\n        :param data: training data, a pandas dataframe with Td rows,\n               and 2 columns, with column \'ds\' indicating date and column \'y\' indicating value\n               and Td is the time dimension\n        :param cross_validation: bool, if the eval result comes from cross_validation.\n               The value is set to True by default. Setting this option to False to\n               speed up the process.\n        :param expect_horizon: int, validation data will be automatically splited from training\n               data, and expect_horizon is the horizon you may need to use once the mode is fitted.\n               The value defaults to None, where 10% of training data will be taken\n               as the validation data.\n        :param freq: the freqency of the training dataframe. the frequency can be anything from the\n               pandas list of frequency strings here:\n               https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliasesDefaulted\n               to None, where an unreliable frequency will be infer implicitly.\n        :param metric_threshold: a trial will be terminated when metric threshold is met\n        :param n_sampling: Number of trials to evaluate in total. Defaults to 16.\n               If hp.grid_search is in search_space, the grid will be run n_sampling of trials\n               and round up n_sampling according to hp.grid_search.\n               If this is -1, (virtually) infinite samples are generated\n               until a stopping condition is met.\n        :param search_alg: str, all supported searcher provided by ray tune\n               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",\n               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and\n               "sigopt")\n        :param search_alg_params: extra parameters for searcher algorithm besides search_space,\n               metric and searcher mode\n        :param scheduler: str, all supported scheduler provided by ray tune\n        :param scheduler_params: parameters for scheduler\n        '
        from bigdl.nano.utils.common import invalidInputError
        if expect_horizon is None:
            expect_horizon = int(0.1 * len(data))
        if freq is None:
            invalidInputError(len(data) >= 2, 'The training dataframe should contains more than 2 records.')
            invalidInputError(pd.api.types.is_datetime64_any_dtype(data['ds'].dtypes), "The 'ds' col should be in datetime 64 type, or you need to set `freq` in fit.")
            self._freq = data['ds'].iloc[1] - data['ds'].iloc[0]
        else:
            self._freq = pd.Timedelta(freq)
        expect_horizon_str = str(self._freq * expect_horizon)
        self.search_space.update({'expect_horizon': expect_horizon_str, 'cross_validation': cross_validation})
        train_data = data if cross_validation else data[:len(data) - expect_horizon]
        validation_data = None if cross_validation else data[len(data) - expect_horizon:]
        n_sampling = recalculate_n_sampling(self.search_space, n_sampling) if n_sampling != -1 else -1
        self.auto_est.fit(data=train_data, validation_data=validation_data, metric=self.metric, metric_mode=self.metric_mode, metric_threshold=metric_threshold, n_sampling=n_sampling, search_space=self.search_space, search_alg=search_alg, search_alg_params=search_alg_params, scheduler=scheduler, scheduler_params=scheduler_params)
        self.best_model = ProphetBuilder().build(self.auto_est.get_best_config())
        self.best_model.model.fit(data)

    def predict(self, horizon=1, freq='D', ds_data=None):
        if False:
            i = 10
            return i + 15
        '\n        Predict using the best model after HPO.\n\n        :param horizon: the number of steps forward to predict\n        :param freq: the freqency of the predicted dataframe, defaulted to day("D"),\n               the frequency can be anything from the pandas list of frequency strings here:\n               https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases\n        :param ds_data: a dataframe that has 1 column \'ds\' indicating date.\n        '
        from bigdl.nano.utils.common import invalidInputError
        if self.best_model.model is None:
            invalidInputError(False, 'You must call fit or restore first before calling predict!')
        return self.best_model.predict(horizon=horizon, freq=freq, ds_data=ds_data)

    def evaluate(self, data, metrics=['mse']):
        if False:
            return 10
        "\n        Evaluate using the best model after HPO.\n\n        :param data: evaluation data, a pandas dataframe with Td rows,\n            and 2 columns, with column 'ds' indicating date and column 'y' indicating value\n            and Td is the time dimension\n        :param metrics: list of string or callable. e.g. ['mse'] or [customized_metrics]\n               If callable function, it signature should be func(y_true, y_pred), where y_true and\n               y_pred are numpy ndarray. The function should return a float value as evaluation\n               result.\n        "
        from bigdl.nano.utils.common import invalidInputError
        if data is None:
            invalidInputError(False, 'Input invalid data of None')
        if self.best_model.model is None:
            invalidInputError(False, 'You must call fit or restore first before calling evaluate!')
        return self.best_model.evaluate(target=data, metrics=metrics)

    def save(self, checkpoint_file):
        if False:
            return 10
        '\n        Save the best model after HPO.\n\n        :param checkpoint_file: The location you want to save the best model, should be a json file\n        '
        from bigdl.nano.utils.common import invalidInputError
        if self.best_model.model is None:
            invalidInputError(False, 'You must call fit or restore first before calling save!')
        self.best_model.save(checkpoint_file)

    def restore(self, checkpoint_file):
        if False:
            for i in range(10):
                print('nop')
        '\n        Restore the best model after HPO.\n\n        :param checkpoint_file: The checkpoint file location you want to load the best model.\n        '
        self.best_model.restore(checkpoint_file)

    def get_best_model(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the best Prophet model.\n        '
        return self.best_model.model