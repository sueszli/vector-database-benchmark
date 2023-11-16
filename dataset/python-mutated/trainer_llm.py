import logging
import os
import time
from typing import Callable, Dict, List, Optional, Union
from torch.utils.tensorboard import SummaryWriter
from ludwig.constants import MINIMUM_BATCH_SIZE, TEST, TRAINING, VALIDATION
from ludwig.data.dataset.base import Dataset
from ludwig.distributed.base import DistributedStrategy, LocalStrategy
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.models.llm import LLM
from ludwig.models.predictor import LlmFineTunePredictor, LlmPredictor
from ludwig.modules.metric_modules import get_initial_validation_value
from ludwig.schema.trainer import BaseTrainerConfig, FineTuneTrainerConfig, NoneTrainerConfig
from ludwig.trainers.base import BaseTrainer
from ludwig.trainers.registry import register_llm_ray_trainer, register_llm_trainer
from ludwig.trainers.trainer import Trainer
from ludwig.types import ModelConfigDict
from ludwig.utils import time_utils
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.metrics_printed_table import print_metrics_table
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.torch_utils import get_torch_device
from ludwig.utils.trainer_utils import append_metrics, get_new_progress_tracker, ProgressTracker
logger = logging.getLogger(__name__)

@register_llm_trainer('none')
@register_llm_ray_trainer('none')
class NoneTrainer(BaseTrainer):
    """NoneTrainer is a trainer that does not train a model, only runs evaluation."""

    def __init__(self, config: NoneTrainerConfig, model: LLM, resume: float=False, skip_save_model: bool=False, skip_save_progress: bool=False, skip_save_log: bool=False, callbacks: List=None, report_tqdm_to_ray=False, random_seed: float=default_random_seed, distributed: Optional[DistributedStrategy]=None, device: Optional[str]=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        :param config: `ludwig.schema.trainer.NoneTrainerConfig` instance that specifies training hyperparameters\n        (default: `ludwig.schema.trainer.NoneTrainerConfig()`).\n        :param model: Underlying Ludwig model\n        :type model: `ludwig.models.llm.LLM`\n        :param resume: Resume training a model that was being trained. (default: False).\n        :type resume: Boolean\n        :param skip_save_model: Disables saving model weights and hyperparameters each time the model improves. By\n                default Ludwig saves model weights after each round of evaluation the validation metric (improves, but\n                if the model is really big that can be time consuming. If you do not want to keep the weights and just\n                find out what performance a model can get with a set of hyperparameters, use this parameter to skip it,\n                but the model will not be loadable later on. (default: False).\n        :type skip_save_model: Boolean\n        :param skip_save_progress: Disables saving progress each round of evaluation. By default Ludwig saves weights\n                and stats after each round of evaluation for enabling resuming of training, but if the model is really\n                big that can be time consuming and will uses twice as much space, use this parameter to skip it, but\n                training cannot be resumed later on. (default: False).\n        :type skip_save_progress: Boolean\n        :param skip_save_log: Disables saving TensorBoard logs. By default Ludwig saves logs for the TensorBoard, but if\n                it is not needed turning it off can slightly increase the overall speed. (default: False).\n        :type skip_save_log: Boolean\n        :param callbacks: List of `ludwig.callbacks.Callback` objects that provide hooks into the Ludwig pipeline.\n                (default: None).\n        :type callbacks: list\n        :param report_tqdm_to_ray: Enables using the ray based tqdm Callback for progress bar reporting\n        :param random_seed: Default initialization for the random seeds (default: 42).\n        :type random_seed: Float\n        :param distributed: Distributed strategy (default: None).\n        :type distributed: `DistributedStrategy`\n        :param device: Device to load the model on from a saved checkpoint (default: None).\n        :type device: str\n        '
        super().__init__()
        self.config = config
        self.distributed = distributed if distributed is not None else LocalStrategy()
        self.skip_save_log = skip_save_log
        self.resume = resume
        self.skip_save_model = skip_save_model
        self.skip_save_progress = skip_save_progress
        self.random_seed = random_seed
        self.callbacks = callbacks or []
        self.report_tqdm_to_ray = report_tqdm_to_ray
        self.device = device if device is not None else get_torch_device()
        self.model = model.to_device(self.device)
        self.model.metrics_to_device(self.device)
        self.model.eval()
        self.batch_size = self.config.batch_size
        self.eval_batch_size = self.config.eval_batch_size
        self.base_learning_rate = self.config.base_learning_rate
        self.should_shuffle = self.config.should_shuffle
        self.epochs = self.config.epochs
        self.train_steps = self.config.train_steps
        self.steps_per_checkpoint = self.config.steps_per_checkpoint
        self.checkpoints_per_epoch = self.config.checkpoints_per_epoch
        self.early_stop = self.config.early_stop
        self.evaluate_training_set = self.config.evaluate_training_set
        self.skip_all_evaluation = self.config.skip_all_evaluation

    def close_writers(self, progress_tracker, save_path, train_summary_writer, validation_summary_writer, test_summary_writer):
        if False:
            i = 10
            return i + 15
        self.callback(lambda c: c.on_trainer_train_teardown(self, progress_tracker, save_path, self.is_coordinator()), coordinator_only=False)
        if train_summary_writer is not None:
            train_summary_writer.close()
        if validation_summary_writer is not None:
            validation_summary_writer.close()
        if test_summary_writer is not None:
            test_summary_writer.close()

    def train(self, training_set: Dataset, validation_set: Optional[Dataset]=None, test_set: Optional[Dataset]=None, save_path: str='model', return_state_dict: bool=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        output_features = self.model.output_features
        tensorboard_log_dir = None
        if self.is_coordinator():
            os.makedirs(save_path, exist_ok=True)
            tensorboard_log_dir = os.path.join(save_path, 'logs')
        self.callback(lambda c: c.on_trainer_train_setup(self, save_path, self.is_coordinator()), coordinator_only=False)
        train_summary_writer = None
        validation_summary_writer = None
        test_summary_writer = None
        if self.is_coordinator() and (not self.skip_save_log) and tensorboard_log_dir:
            train_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, TRAINING))
            if validation_set is not None and validation_set.size > 0:
                validation_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, VALIDATION))
            if test_set is not None and test_set.size > 0:
                test_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, TEST))
        set_random_seed(self.random_seed)
        progress_tracker = get_new_progress_tracker(batch_size=self.batch_size, learning_rate=self.base_learning_rate, best_eval_metric_value=get_initial_validation_value(self.validation_metric), best_increase_batch_size_eval_metric=float('inf'), output_features=output_features)
        return_value = self.model if not return_state_dict else self.model.cpu().state_dict()
        if self.skip_all_evaluation:
            self.close_writers(progress_tracker, save_path, train_summary_writer, validation_summary_writer, test_summary_writer)
            return (return_value, progress_tracker.train_metrics, progress_tracker.validation_metrics, progress_tracker.test_metrics)
        try:
            self.run_evaluation(training_set, validation_set, test_set, progress_tracker, train_summary_writer, validation_summary_writer, test_summary_writer, output_features, save_path)
        finally:
            self.close_writers(progress_tracker, save_path, train_summary_writer, validation_summary_writer, test_summary_writer)
        return (return_value, progress_tracker.train_metrics, progress_tracker.validation_metrics, progress_tracker.test_metrics)

    def train_online(self, dataset):
        if False:
            return 10
        pass

    def tune_batch_size(self, config: ModelConfigDict, training_set: Dataset, random_seed: int=default_random_seed, max_trials: int=20, halving_limit: int=3, snapshot_weights: bool=True, on_best_batch_size_updated: Optional[Callable[[int, float, int], None]]=None, tune_for_training: bool=True) -> int:
        if False:
            for i in range(10):
                print('nop')
        return MINIMUM_BATCH_SIZE

    @property
    def validation_field(self):
        if False:
            while True:
                i = 10
        return self.config.validation_field

    @property
    def validation_metric(self):
        if False:
            return 10
        return self.config.validation_metric

    def shutdown(self):
        if False:
            print('Hello World!')
        pass

    @property
    def local_rank(self) -> int:
        if False:
            i = 10
            return i + 15
        return 0

    def barrier(self):
        if False:
            return 10
        pass

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        self.shutdown()

    @staticmethod
    def get_schema_cls() -> BaseTrainerConfig:
        if False:
            for i in range(10):
                print('nop')
        return NoneTrainerConfig

    def is_coordinator(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.distributed.rank() == 0

    def callback(self, fn, coordinator_only=True):
        if False:
            i = 10
            return i + 15
        if not coordinator_only or self.is_coordinator():
            for callback in self.callbacks:
                fn(callback)

    def evaluation(self, dataset: 'Dataset', dataset_name: str, metrics_log: Dict[str, Dict[str, List[TrainerMetric]]], batch_size: int, progress_tracker: ProgressTracker):
        if False:
            i = 10
            return i + 15
        predictor = LlmPredictor(self.model, batch_size=batch_size, distributed=self.distributed, report_tqdm_to_ray=self.report_tqdm_to_ray)
        (metrics, _) = predictor.batch_evaluation(dataset, collect_predictions=False, dataset_name=dataset_name)
        return append_metrics(self.model, dataset_name, metrics, metrics_log, progress_tracker)

    @classmethod
    def write_eval_summary(cls, summary_writer, metrics, step):
        if False:
            while True:
                i = 10
        if not summary_writer:
            return
        for (feature_name, output_feature) in metrics.items():
            for (metric_name, metrics) in output_feature.items():
                if metrics:
                    metric_tag = f'{feature_name}/epoch_{metric_name}'
                    metric_val = metrics[-1][-1]
                    summary_writer.add_scalar(metric_tag, metric_val, global_step=step)
        summary_writer.flush()

    def run_evaluation(self, training_set: Union['Dataset', 'RayDataset'], validation_set: Optional[Union['Dataset', 'RayDataset']], test_set: Optional[Union['Dataset', 'RayDataset']], progress_tracker: ProgressTracker, train_summary_writer: SummaryWriter, validation_summary_writer: SummaryWriter, test_summary_writer: SummaryWriter, output_features: LudwigFeatureDict, save_path: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Runs evaluation over training, validation, and test sets.\n\n        Also:\n        - Prints results, saves results to the progress tracker.\n        - Saves the model if the validation score is the best so far\n        - If there is no validation set, the model is always saved.\n\n        Returns whether the trainer should early stop, based on validation metrics history.\n        '
        start_time = time.time()
        self.callback(lambda c: c.on_eval_start(self, progress_tracker, save_path))
        progress_tracker.checkpoint_number += 1
        if self.is_coordinator():
            logger.info(f'\nRunning evaluation for step: {progress_tracker.steps}, epoch: {progress_tracker.epoch}')
        if self.evaluate_training_set:
            self.evaluation(training_set, 'train', progress_tracker.train_metrics, self.eval_batch_size, progress_tracker)
        self.write_eval_summary(summary_writer=train_summary_writer, metrics=progress_tracker.train_metrics, step=progress_tracker.steps)
        if validation_set is not None:
            self.callback(lambda c: c.on_validation_start(self, progress_tracker, save_path))
            self.evaluation(validation_set, VALIDATION, progress_tracker.validation_metrics, self.eval_batch_size, progress_tracker)
            self.write_eval_summary(summary_writer=validation_summary_writer, metrics=progress_tracker.validation_metrics, step=progress_tracker.steps)
            self.callback(lambda c: c.on_validation_end(self, progress_tracker, save_path))
        if test_set is not None:
            self.callback(lambda c: c.on_test_start(self, progress_tracker, save_path))
            self.evaluation(test_set, TEST, progress_tracker.test_metrics, self.eval_batch_size, progress_tracker)
            self.write_eval_summary(summary_writer=test_summary_writer, metrics=progress_tracker.test_metrics, step=progress_tracker.steps)
            self.callback(lambda c: c.on_test_end(self, progress_tracker, save_path))
        elapsed_time = (time.time() - start_time) * 1000.0
        if self.is_coordinator():
            logger.info(f'Evaluation took {time_utils.strdelta(elapsed_time)}\n')
            print_metrics_table(output_features, progress_tracker.train_metrics, progress_tracker.validation_metrics, progress_tracker.test_metrics)
        self.callback(lambda c: c.on_eval_end(self, progress_tracker, save_path))
        return False

@register_llm_trainer('finetune')
class FineTuneTrainer(Trainer):

    @staticmethod
    def get_schema_cls():
        if False:
            return 10
        return FineTuneTrainerConfig

    def __init__(self, config: FineTuneTrainerConfig, model: LLM, resume: float=False, skip_save_model: bool=False, skip_save_progress: bool=False, skip_save_log: bool=False, callbacks: List=None, report_tqdm_to_ray=False, random_seed: float=default_random_seed, distributed: Optional[DistributedStrategy]=None, device: Optional[str]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, model, resume, skip_save_model, skip_save_progress, skip_save_log, callbacks, report_tqdm_to_ray, random_seed, distributed, device, **kwargs)

    def evaluation(self, dataset, dataset_name, metrics_log, batch_size, progress_tracker):
        if False:
            while True:
                i = 10
        predictor = LlmFineTunePredictor(self.model, batch_size=batch_size, distributed=self.distributed, report_tqdm_to_ray=self.report_tqdm_to_ray)
        (metrics, _) = predictor.batch_evaluation(dataset, collect_predictions=False, dataset_name=dataset_name)
        return append_metrics(self.model, dataset_name, metrics, metrics_log, progress_tracker)

class RemoteLLMTrainer(NoneTrainer):

    def __init__(self, gpus=None, gpu_memory_limit=None, allow_parallel_threads=True, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.train = self.distributed.return_first(self.train)
        self.train_online = self.distributed.return_first(self.train_online)

class RemoteLLMFineTuneTrainer(FineTuneTrainer):

    def __init__(self, gpus=None, gpu_memory_limit=None, allow_parallel_threads=True, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.train = self.distributed.return_first(self.train)
        self.train_online = self.distributed.return_first(self.train_online)