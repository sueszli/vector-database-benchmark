import logging
from collections import defaultdict
from typing import Dict, List, Tuple, TYPE_CHECKING
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUTO, COMBINED, LOSS
from ludwig.models.base import BaseModel
from ludwig.modules.metric_modules import get_best_function
from ludwig.utils.data_utils import save_json
from ludwig.utils.metric_utils import TrainerMetric
if TYPE_CHECKING:
    from ludwig.features.base_feature import OutputFeature
    from ludwig.schema.trainer import BaseTrainerConfig
logger = logging.getLogger(__name__)

@DeveloperAPI
def initialize_trainer_metric_dict(output_features) -> Dict[str, Dict[str, List[TrainerMetric]]]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a dict of dict of metrics, output_feature_name -> metric_name -> List[TrainerMetric].'
    metrics = defaultdict(lambda : defaultdict(list))
    return metrics

def get_latest_metrics_dict(progress_tracker_metrics: Dict[str, Dict[str, List[TrainerMetric]]]) -> Dict[str, Dict[str, float]]:
    if False:
        while True:
            i = 10
    'Returns a dict of field name -> metric name -> latest metric value.'
    latest_metrics_dict = defaultdict(dict)
    for (feature_name, metrics_dict) in progress_tracker_metrics.items():
        for (metric_name, metrics) in metrics_dict.items():
            if metrics:
                latest_metrics_dict[feature_name][metric_name] = metrics[-1][-1]
    return latest_metrics_dict

@DeveloperAPI
def get_new_progress_tracker(batch_size: int, best_eval_metric_value: float, best_increase_batch_size_eval_metric: float, learning_rate: float, output_features: Dict[str, 'OutputFeature']):
    if False:
        return 10
    'Returns a new instance of a ProgressTracker with empty metrics.'
    return ProgressTracker(epoch=0, batch_size=batch_size, steps=0, tune_checkpoint_num=0, checkpoint_number=0, best_eval_metric_steps=0, best_eval_metric_epoch=0, best_eval_metric_checkpoint_number=0, last_learning_rate_reduction_steps=0, last_increase_batch_size_steps=0, last_improvement_steps=0, best_eval_metric_value=best_eval_metric_value, best_increase_batch_size_eval_metric=best_increase_batch_size_eval_metric, last_increase_batch_size_eval_metric_improvement=0, learning_rate=learning_rate, num_reductions_learning_rate=0, num_increases_batch_size=0, train_metrics=initialize_trainer_metric_dict(output_features), validation_metrics=initialize_trainer_metric_dict(output_features), test_metrics=initialize_trainer_metric_dict(output_features), last_learning_rate_reduction=0, last_increase_batch_size=0, best_eval_train_metrics={}, best_eval_validation_metrics={}, best_eval_test_metrics={})

@DeveloperAPI
class ProgressTracker:

    def __init__(self, epoch: int, batch_size: int, steps: int, tune_checkpoint_num: int, checkpoint_number: int, best_eval_metric_steps: int, best_eval_metric_epoch: int, best_eval_metric_checkpoint_number: int, last_improvement_steps: int, last_learning_rate_reduction_steps: int, last_increase_batch_size_steps: int, best_eval_metric_value: float, best_increase_batch_size_eval_metric: float, last_increase_batch_size_eval_metric_improvement: int, learning_rate: float, num_reductions_learning_rate: int, num_increases_batch_size: int, train_metrics: Dict[str, Dict[str, List[TrainerMetric]]], validation_metrics: Dict[str, Dict[str, List[TrainerMetric]]], test_metrics: Dict[str, Dict[str, List[TrainerMetric]]], last_learning_rate_reduction: int, last_increase_batch_size: int, best_eval_train_metrics: Dict[str, Dict[str, float]], best_eval_validation_metrics: Dict[str, Dict[str, float]], best_eval_test_metrics: Dict[str, Dict[str, float]]):
        if False:
            print('Hello World!')
        'JSON-serializable holder object that stores information related to training progress.\n\n        [train/vali/test]_metrics is a nested dictionary of TrainerMetrics: feature_name -> metric_name ->\n        List[TrainerMetrics], with one entry per training checkpoint.\n\n        Note that when a model resumes training from a checkpoint, the progress tracker is deserialized from JSON, which\n        automatically converts TrainerMetrics namedtuples into regular (epoch, steps, value) tuples.\n\n        Args:\n            epoch: The current epoch number.\n            steps: The current step of training.\n            batch_size: The current batch size.\n            tune_checkpoint_num: The hyperopt checkpoint number (Ray Tune).\n            checkpoint_number: The current checkpoint number.\n\n            best_eval_metric_steps: The step of training that has the best evaluation so far.\n            best_eval_metric_epoch: The epoch of training that has the best evaluation so far.\n            best_eval_metric_checkpoint_number: The checkpoint number that has the best evaluation so far.\n\n            last_improvement_steps: The number of steps since the last improvement.\n            last_learning_rate_reduction_steps: The training step of the last learning rate reduction.\n            last_increase_batch_size_steps: The training_step of the the last batch size increase.\n\n            best_eval_metric_value: The metric value of the best evaluation so far.\n            best_increase_batch_size_eval_metric:\n                The metric value of the best evaluation so far, for increasing the batch size.\n\n            last_learning_rate_reduction: The number of steps since the last learning rate reduction.\n            last_increase_batch_size: The number of steps since the last batch size increase.\n\n            last_increase_batch_size_eval_metric_improvement:\n                The number of checkpoints since the last batch size increase.\n\n            num_reductions_learning_rate: The number of total reductions in learning rate.\n            num_increases_batch_size: The number of total increases in batch size.\n\n            train_metrics: Training metrics. <output feature name> -> <metric name> -> History of metrics.\n            validation_metrics: Validation metrics. <output feature name> -> <metric name> -> History of metrics.\n            test_metrics: Test metrics. <output feature name> -> <metric name> -> History of metrics.\n\n            best_eval_train_metrics:\n                Best eval train metrics: <output feature name> -> <metric name> -> <metric value>.\n            best_eval_validation_metrics:\n                Best eval validation metrics: <output feature name> -> <metric name> -> <metric value>.\n            best_eval_test_metrics:\n                Best eval test metrics: <output feature name> -> <metric name> -> <metric value>.\n        '
        self.batch_size = batch_size
        self.epoch = epoch
        self.steps = steps
        self.tune_checkpoint_num = tune_checkpoint_num
        self.checkpoint_number = checkpoint_number
        self.best_eval_metric_steps = best_eval_metric_steps
        self.best_eval_metric_epoch = best_eval_metric_epoch
        self.best_eval_metric_checkpoint_number = best_eval_metric_checkpoint_number
        self.last_improvement_steps = last_improvement_steps
        self.last_learning_rate_reduction_steps = last_learning_rate_reduction_steps
        self.last_learning_rate_reduction = last_learning_rate_reduction
        self.last_increase_batch_size_steps = last_increase_batch_size_steps
        self.last_increase_batch_size = last_increase_batch_size
        self.learning_rate = learning_rate
        self.best_eval_metric_value = best_eval_metric_value
        self.best_increase_batch_size_eval_metric = best_increase_batch_size_eval_metric
        self.last_increase_batch_size_eval_metric_improvement = last_increase_batch_size_eval_metric_improvement
        self.num_reductions_learning_rate = num_reductions_learning_rate
        self.num_increases_batch_size = num_increases_batch_size
        self.train_metrics = train_metrics
        self.validation_metrics = validation_metrics
        self.test_metrics = test_metrics
        self.best_eval_train_metrics = best_eval_train_metrics
        self.best_eval_validation_metrics = best_eval_validation_metrics
        self.best_eval_test_metrics = best_eval_test_metrics

    def save(self, filepath):
        if False:
            for i in range(10):
                print('nop')
        save_json(filepath, self.__dict__)

    @staticmethod
    def load(progress_tracking_dict: Dict):
        if False:
            for i in range(10):
                print('nop')
        from ludwig.utils.backward_compatibility import upgrade_model_progress
        loaded = upgrade_model_progress(progress_tracking_dict)
        return ProgressTracker(**loaded)

    def log_metrics(self):
        if False:
            while True:
                i = 10
        log_metrics = {'batch_size': self.batch_size, 'epoch': self.epoch, 'steps': self.steps, 'tune_checkpoint_num': self.tune_checkpoint_num, 'checkpoint_number': self.checkpoint_number, 'last_improvement_steps': self.last_improvement_steps, 'best_eval_metric_steps': self.best_eval_metric_steps, 'best_eval_metric_epoch': self.best_eval_metric_epoch, 'best_eval_metric_checkpoint_number': self.best_eval_metric_checkpoint_number, 'learning_rate': self.learning_rate, 'best_valid_metric': self.best_eval_metric_value, 'num_reductions_lr': self.num_reductions_learning_rate, 'num_increases_bs': self.num_increases_batch_size}
        for metrics_dict_name in ['train_metrics', 'validation_metrics', 'test_metrics']:
            metrics_dict = getattr(self, metrics_dict_name)
            for feature_name in metrics_dict:
                for (metric_name, metrics_tuples) in metrics_dict[feature_name].items():
                    if metrics_tuples:
                        log_metrics[f'{metrics_dict_name}.{feature_name}.{metric_name}'] = metrics_tuples[-1][-1]
        for (feature_name, metrics) in self.best_eval_train_metrics.items():
            for (metric_name, metric_value) in metrics.items():
                log_metrics[f'best.train_metrics.{feature_name}.{metric_name}'] = metric_value
        for (feature_name, metrics) in self.best_eval_validation_metrics.items():
            for (metric_name, metric_value) in metrics.items():
                log_metrics[f'best.validation_metrics.{feature_name}.{metric_name}'] = metric_value
        for (feature_name, metrics) in self.best_eval_test_metrics.items():
            for (metric_name, metric_value) in metrics.items():
                log_metrics[f'best.test_metrics.{feature_name}.{metric_name}'] = metric_value
        return log_metrics

@DeveloperAPI
def append_metrics(model: BaseModel, dataset_name: Literal['train', 'validation', 'test'], results: Dict[str, Dict[str, float]], metrics_log: Dict[str, Dict[str, List[TrainerMetric]]], progress_tracker: ProgressTracker) -> Dict[str, Dict[str, List[TrainerMetric]]]:
    if False:
        for i in range(10):
            print('nop')
    epoch = progress_tracker.epoch
    steps = progress_tracker.steps
    for output_feature in model.output_features:
        scores = [dataset_name]
        metric_names = sorted(results[output_feature].keys())
        for metric in metric_names:
            if metric in results[output_feature]:
                score = results[output_feature][metric]
                metrics_log[output_feature][metric].append(TrainerMetric(epoch=epoch, step=steps, value=score))
                scores.append(score)
    metrics_log[COMBINED][LOSS].append(TrainerMetric(epoch=epoch, step=steps, value=results[COMBINED][LOSS]))
    return metrics_log

@DeveloperAPI
def get_total_steps(epochs: int, steps_per_epoch: int, train_steps: int):
    if False:
        return 10
    'Returns train_steps if provided, otherwise epochs * steps_per_epoch.'
    if train_steps:
        return train_steps
    return epochs * steps_per_epoch

@DeveloperAPI
def get_final_steps_per_checkpoint(steps_per_epoch: int, steps_per_checkpoint: int=0, checkpoints_per_epoch: float=0, should_log: bool=False):
    if False:
        return 10
    'Returns the steps per checkpoint to use for the training loop, given user+default inputs.'
    if steps_per_checkpoint != 0 and checkpoints_per_epoch != 0:
        raise ValueError('It is invalid to specify both checkpoints_per_epoch AND steps_per_checkpoint. Please specify one or the other, or specify neither to checkpoint/eval the model every epoch.')
    if checkpoints_per_epoch != 0:
        steps_per_checkpoint = int(steps_per_epoch / checkpoints_per_epoch)
    if steps_per_checkpoint > steps_per_epoch:
        if should_log:
            logger.info(f'Note: steps_per_checkpoint (was {steps_per_checkpoint}) is now set to the number of steps per epoch: {steps_per_epoch}.\n')
        return steps_per_epoch
    if steps_per_checkpoint == 0:
        return steps_per_epoch
    return steps_per_checkpoint

@DeveloperAPI
def get_training_report(validation_field: str, validation_metric: str, include_test_set: bool, train_valiset_stats: Dict[str, Dict[str, List[float]]], train_testset_stats: Dict[str, Dict[str, List[float]]]) -> List[Tuple[str, str]]:
    if False:
        i = 10
        return i + 15
    'Returns a training report in the form of a list [(report item, value)].'
    validation_field_result = train_valiset_stats[validation_field]
    best_function = get_best_function(validation_metric)
    training_report = []
    (best_vali_index, (epoch_best_validation_metric, step_best_validation_metric, best_validation_metric)) = best_function(enumerate(validation_field_result[validation_metric]), key=lambda index_epoch_step_value: index_epoch_step_value[1][-1])
    training_report.append(['Validation feature', validation_field])
    training_report.append(['Validation metric', validation_metric])
    training_report.append(['Best model step', step_best_validation_metric])
    training_report.append(['Best model epoch', epoch_best_validation_metric + 1])
    training_report.append([f"Best model's validation {validation_metric}", best_validation_metric])
    if include_test_set:
        validation_selected_test_metric_score = train_testset_stats[validation_field][validation_metric][best_vali_index][-1]
        training_report.append([f"Best model's test {validation_metric}", validation_selected_test_metric_score])
    return training_report

def get_rendered_batch_size_grad_accum(config: 'BaseTrainerConfig', num_workers: int) -> Tuple[int, int]:
    if False:
        print('Hello World!')
    'Returns the batch size and gradient accumulation steps to use for training.\n\n    For batch_size==AUTO:\n    1. effective_batch_size is not AUTO and gradient_accumulation_steps is not AUTO:\n        batch size is set to the effective batch size divided by the gradient accumulation steps, divided by the\n        number of workers.\n    2. effective_batch_size is AUTO or gradient_accumulation_steps is AUTO:\n        batch size remains AUTO.\n\n    For gradient_accumulation_steps==AUTO:\n    1. batch size is AUTO:\n        gradient accumulation steps remains AUTO.\n    2. batch_size is not AUTO and effective batch size is not AUTO:\n        gradient accumulation steps is set to the effective batch size divided by the batch size, divided by the number\n        of workers.\n    3. batch size is not AUTO and effective batch size is AUTO:\n        gradient accumulation steps is set to 1.\n    '
    effective_batch_size = config.effective_batch_size
    batch_size = config.batch_size
    gradient_accumulation_steps = config.gradient_accumulation_steps
    if config.batch_size == AUTO:
        if config.effective_batch_size != AUTO and config.gradient_accumulation_steps != AUTO:
            batch_size = max(int(effective_batch_size / gradient_accumulation_steps / num_workers), 1)
    if config.gradient_accumulation_steps == AUTO:
        if config.batch_size != AUTO:
            if config.effective_batch_size != AUTO:
                gradient_accumulation_steps = max(int(effective_batch_size / batch_size / num_workers), 1)
            else:
                gradient_accumulation_steps = 1
    return (batch_size, gradient_accumulation_steps)