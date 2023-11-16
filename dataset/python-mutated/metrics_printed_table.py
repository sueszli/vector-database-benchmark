import logging
from typing import Dict, List
from tabulate import tabulate
from ludwig.constants import COMBINED, LOSS
from ludwig.utils.metric_utils import TrainerMetric
logger = logging.getLogger(__name__)

def get_metric_value_or_empty(metrics_log: Dict[str, List[TrainerMetric]], metric_name: str):
    if False:
        while True:
            i = 10
    'Returns the metric value if it exists or empty.'
    if metric_name not in metrics_log:
        return ''
    return metrics_log[metric_name][-1][-1]

def print_table_for_single_output_feature(train_metrics_log: Dict[str, List[TrainerMetric]], validation_metrics_log: Dict[str, List[TrainerMetric]], test_metrics_log: Dict[str, List[TrainerMetric]], combined_loss_for_each_split: List[float]) -> None:
    if False:
        return 10
    'Prints the metrics table for a single output feature.\n\n    Args:\n        train_metrics_log: Dict from metric name to list of TrainerMetric.\n        validation_metrics_log: Dict from metric name to list of TrainerMetric.\n        test_metrics_log: Dict from metric name to list of TrainerMetric.\n    '
    all_metric_names = set()
    all_metric_names.update(train_metrics_log.keys())
    all_metric_names.update(validation_metrics_log.keys())
    all_metric_names.update(test_metrics_log.keys())
    all_metric_names = sorted(list(all_metric_names))
    printed_table = [['train', 'validation', 'test']]
    for metric_name in all_metric_names:
        metrics_for_each_split = [get_metric_value_or_empty(train_metrics_log, metric_name), get_metric_value_or_empty(validation_metrics_log, metric_name), get_metric_value_or_empty(test_metrics_log, metric_name)]
        printed_table.append([metric_name] + metrics_for_each_split)
    printed_table.append(['combined_loss'] + combined_loss_for_each_split)
    logger.info(tabulate(printed_table, headers='firstrow', tablefmt='fancy_grid', floatfmt='.4f'))

def print_metrics_table(output_features: Dict[str, 'OutputFeature'], train_metrics_log: Dict[str, Dict[str, List[TrainerMetric]]], validation_metrics_log: Dict[str, Dict[str, List[TrainerMetric]]], test_metrics_log: Dict[str, Dict[str, List[TrainerMetric]]]):
    if False:
        return 10
    'Prints a table of metrics table for each output feature, for each split.\n\n    Example:\n    ╒═══════════════╤═════════╤══════════════╤════════╕\n    │               │   train │   validation │   test │\n    ╞═══════════════╪═════════╪══════════════╪════════╡\n    │ accuracy      │  0.8157 │       0.6966 │ 0.8090 │\n    ├───────────────┼─────────┼──────────────┼────────┤\n    │ loss          │  0.4619 │       0.5039 │ 0.4488 │\n    ├───────────────┼─────────┼──────────────┼────────┤\n    │ precision     │  0.8274 │       0.6250 │ 0.7818 │\n    ├───────────────┼─────────┼──────────────┼────────┤\n    │ recall        │  0.6680 │       0.4545 │ 0.6615 │\n    ├───────────────┼─────────┼──────────────┼────────┤\n    │ roc_auc       │  0.8471 │       0.7706 │ 0.8592 │\n    ├───────────────┼─────────┼──────────────┼────────┤\n    │ specificity   │  0.9105 │       0.8393 │ 0.8938 │\n    ├───────────────┼─────────┼──────────────┼────────┤\n    │ combined_loss │  0.4619 │       0.5039 │ 0.4488 │\n    ╘═══════════════╧═════════╧══════════════╧════════╛\n    '
    combined_loss_for_each_split = [get_metric_value_or_empty(train_metrics_log[COMBINED], LOSS), get_metric_value_or_empty(validation_metrics_log[COMBINED], LOSS), get_metric_value_or_empty(test_metrics_log[COMBINED], LOSS)]
    for output_feature_name in sorted(output_features.keys()):
        if output_feature_name == COMBINED:
            continue
        print_table_for_single_output_feature(train_metrics_log[output_feature_name], validation_metrics_log[output_feature_name], test_metrics_log[output_feature_name], combined_loss_for_each_split)