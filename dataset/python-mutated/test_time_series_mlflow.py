"""Module to test time_series "MLflow" functionality
"""
from pycaret.time_series import TSForecastingExperiment

def test_mlflow_logging(load_pos_and_neg_data):
    if False:
        while True:
            i = 10
    'Tests the logging of MLFlow experiment'
    data = load_pos_and_neg_data
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, session_id=42, log_experiment=True, experiment_name='ts_unit_test', log_plots=True)
    model = exp.create_model('naive')
    _ = exp.tune_model(model)
    _ = exp.compare_models(include=['naive', 'ets'])
    mlflow_logs = exp.get_logs()
    last_start = mlflow_logs['start_time'].max()
    last_experiment_usi = mlflow_logs.query('start_time == @last_start')['tags.USI'].unique()[0]
    num_create_models = len(mlflow_logs.query("`tags.USI` == @last_experiment_usi & `tags.Source` == 'create_model'"))
    num_tune_models = len(mlflow_logs.query("`tags.USI` == @last_experiment_usi &`tags.Source` == 'tune_model'"))
    num_compare_models = len(mlflow_logs.query("`tags.USI` == @last_experiment_usi &`tags.Source` == 'compare_models'"))
    assert num_create_models == 1
    assert num_tune_models == 1
    assert num_compare_models == 2

def test_mlflow_log_setup(load_pos_and_neg_data):
    if False:
        while True:
            i = 10
    'Tests the logging of MLFlow for plots during setup'
    data = load_pos_and_neg_data
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, session_id=42, log_experiment=True, experiment_name='ts_unit_test', log_plots=True)
    mlflow_logs = exp.get_logs()
    num_setup = len(mlflow_logs.query("`tags.Source` == 'setup'"))
    assert num_setup == 1