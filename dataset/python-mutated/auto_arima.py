import warnings
from bigdl.chronos.model.arima import ARIMABuilder, ARIMAModel
from bigdl.chronos.autots.utils import recalculate_n_sampling

class AutoARIMA:

    def __init__(self, p=2, q=2, seasonal=True, P=1, Q=1, m=7, metric='mse', metric_mode=None, logs_dir='/tmp/auto_arima_logs', cpus_per_trial=1, name='auto_arima', remote_dir=None, load_dir=None, **arima_config):
        if False:
            while True:
                i = 10
        '\n        Create an automated ARIMA Model.\n        User need to specify either the exact value or the search space of\n        the ARIMA model hyperparameters. For details of the ARIMA model hyperparameters, refer to\n        https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA.\n\n        :param p: Int or hp sampling function from an integer space for hyperparameter p\n               of the ARIMA model.\n               For hp sampling, see bigdl.chronos.orca.automl.hp for more details.\n               e.g. hp.randint(0, 3).\n        :param q: Int or hp sampling function from an integer space for hyperparameter q\n               of the ARIMA model.\n               e.g. hp.randint(0, 3).\n        :param seasonal: Bool or hp sampling function from an integer space for whether to add\n               seasonal components to the ARIMA model.\n               e.g. hp.choice([True, False]).\n        :param P: Int or hp sampling function from an integer space for hyperparameter P\n               of the ARIMA model.\n               For hp sampling, see bigdl.chronos.orca.automl.hp for more details.\n               e.g. hp.randint(0, 3).\n        :param Q: Int or hp sampling function from an integer space for hyperparameter Q\n               of the ARIMA model.\n               e.g. hp.randint(0, 3).\n        :param m: Int or hp sampling function from an integer space for hyperparameter p\n               of the ARIMA model.\n               e.g. hp.choice([4, 7, 12, 24, 365]).\n        :param metric: String or customized evaluation metric function.\n            If string, metric is the evaluation metric name to optimize, e.g. "mse".\n            If callable function, it signature should be func(y_true, y_pred), where y_true and\n            y_pred are numpy ndarray. The function should return a float value as evaluation result.\n        :param metric_mode: One of ["min", "max"]. "max" means greater metric value is better.\n            You have to specify metric_mode if you use a customized metric function.\n            You don\'t have to specify metric_mode if you use the built-in metric in\n            bigdl.orca.automl.metrics.Evaluator.\n        :param logs_dir: Local directory to save logs and results. It defaults to\n               "/tmp/auto_arima_logs"\n        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.\n        :param name: name of the AutoARIMA. It defaults to "auto_arima"\n        :param remote_dir: String. Remote directory to sync training results and checkpoints. It\n            defaults to None and doesn\'t take effects while running in local. While running in\n            cluster, it defaults to "hdfs:///tmp/{name}".\n        :param arima_config: Other ARIMA hyperparameters.\n\n        '
        if load_dir:
            self.best_model = ARIMAModel()
            self.best_model.restore(load_dir)
        try:
            from bigdl.orca.automl.auto_estimator import AutoEstimator
            self.search_space = {'p': p, 'q': q, 'seasonal': seasonal, 'P': P, 'Q': Q, 'm': m}
            self.metric = metric
            self.metric_mode = metric_mode
            model_builder = ARIMABuilder()
            self.auto_est = AutoEstimator(model_builder=model_builder, logs_dir=logs_dir, resources_per_trial={'cpu': cpus_per_trial}, remote_dir=remote_dir, name=name)
        except ImportError:
            warnings.warn('You need to install `bigdl-orca[automl]` to use `fit` function.')

    def fit(self, data, epochs=1, validation_data=None, metric_threshold=None, n_sampling=1, search_alg=None, search_alg_params=None, scheduler=None, scheduler_params=None):
        if False:
            i = 10
            return i + 15
        '\n        Automatically fit the model and search for the best hyperparameters.\n\n        :param data: Training data, A 1-D numpy array.\n        :param epochs: Max number of epochs to train in each trial. Defaults to 1.\n               If you have also set metric_threshold, a trial will stop if either it has been\n               optimized to the metric_threshold or it has been trained for {epochs} epochs.\n        :param validation_data: Validation data. A 1-D numpy array.\n        :param metric_threshold: a trial will be terminated when metric threshold is met\n        :param n_sampling: Number of trials to evaluate in total. Defaults to 1.\n               If hp.grid_search is in search_space, the grid will be run n_sampling of trials\n               and round up n_sampling according to hp.grid_search.\n               If this is -1, (virtually) infinite samples are generated\n               until a stopping condition is met.\n        :param search_alg: str, all supported searcher provided by ray tune\n               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",\n               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and\n               "sigopt")\n        :param search_alg_params: extra parameters for searcher algorithm besides search_space,\n               metric and searcher mode\n        :param scheduler: str, all supported scheduler provided by ray tune\n        :param scheduler_params: parameters for scheduler\n        '
        n_sampling = recalculate_n_sampling(self.search_space, n_sampling) if n_sampling != -1 else -1
        self.auto_est.fit(data=data, validation_data=validation_data, metric=self.metric, metric_mode=self.metric_mode, metric_threshold=metric_threshold, n_sampling=n_sampling, search_space=self.search_space, search_alg=search_alg, search_alg_params=search_alg_params, scheduler=scheduler, scheduler_params=scheduler_params)

    def get_best_model(self):
        if False:
            while True:
                i = 10
        '\n        Get the best ARIMA model.\n        '
        return self.auto_est.get_best_model()