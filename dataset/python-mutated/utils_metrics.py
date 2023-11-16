import logging
import torchmetrics
log = logging.getLogger('NP.metrics')
METRICS = {'MAE': ['MeanAbsoluteError', {}], 'MSE': ['MeanSquaredError', {'squared': True}], 'RMSE': ['MeanSquaredError', {'squared': False}]}

def get_metrics(metric_input):
    if False:
        return 10
    '\n    Returns a dict of metrics.\n\n    Parameters\n    ----------\n        metrics : input received from the user\n            List of metrics to use.\n\n    Returns\n    -------\n        dict\n            Dict of names of torchmetrics.Metric metrics\n    '
    if metric_input is None:
        return {}
    elif metric_input is True:
        return {'MAE': METRICS['MAE'], 'RMSE': METRICS['RMSE']}
    elif isinstance(metric_input, str):
        if metric_input.upper() in METRICS.keys():
            return {metric_input: METRICS[metric_input.upper()]}
        else:
            raise ValueError('Received unsupported argument for collect_metrics.')
    elif isinstance(metric_input, list):
        if all([m.upper() in METRICS.keys() for m in metric_input]):
            return {m: METRICS[m.upper()] for m in metric_input}
        else:
            raise ValueError('Received unsupported argument for collect_metrics.')
    elif isinstance(metric_input, dict):
        try:
            for _metric in metric_input.values():
                torchmetrics.__dict__[_metric]()
        except KeyError:
            raise ValueError('Received unsupported argument for collect_metrics.All metrics must be valid names of torchmetrics.Metric objects.')
        return {k: [v, {}] for (k, v) in metric_input.items()}
    elif metric_input is not False:
        raise ValueError('Received unsupported argument for collect_metrics.')