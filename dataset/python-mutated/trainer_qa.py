"""
A subclass of `Trainer` specific to Question-Answering tasks
"""
import math
import time
from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput, speed_metrics
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

class QuestionAnsweringTrainer(Trainer):

    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str='eval'):
        if False:
            while True:
                i = 10
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        start_time = time.time()
        try:
            output = eval_loop(eval_dataloader, description='Evaluation', prediction_loss_only=True if compute_metrics is None else None, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        finally:
            self.compute_metrics = compute_metrics
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f'{metric_key_prefix}_jit_compilation_time' in output.metrics:
            start_time += output.metrics[f'{metric_key_prefix}_jit_compilation_time']
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, num_samples=output.num_samples, num_steps=math.ceil(output.num_samples / total_batch_size)))
        if self.post_process_function is not None and self.compute_metrics is not None and self.args.should_save:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)
            for key in list(metrics.keys()):
                if not key.startswith(f'{metric_key_prefix}_'):
                    metrics[f'{metric_key_prefix}_{key}'] = metrics.pop(key)
            metrics.update(output.metrics)
        else:
            metrics = output.metrics
        if self.args.should_log:
            self.log(metrics)
        if self.args.tpu_metrics_debug or self.args.debug:
            xm.master_print(met.metrics_report())
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str='test'):
        if False:
            i = 10
            return i + 15
        predict_dataloader = self.get_test_dataloader(predict_dataset)
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        start_time = time.time()
        try:
            output = eval_loop(predict_dataloader, description='Prediction', prediction_loss_only=True if compute_metrics is None else None, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        finally:
            self.compute_metrics = compute_metrics
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f'{metric_key_prefix}_jit_compilation_time' in output.metrics:
            start_time += output.metrics[f'{metric_key_prefix}_jit_compilation_time']
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, num_samples=output.num_samples, num_steps=math.ceil(output.num_samples / total_batch_size)))
        if self.post_process_function is None or self.compute_metrics is None:
            return output
        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, 'predict')
        metrics = self.compute_metrics(predictions)
        for key in list(metrics.keys()):
            if not key.startswith(f'{metric_key_prefix}_'):
                metrics[f'{metric_key_prefix}_{key}'] = metrics.pop(key)
        metrics.update(output.metrics)
        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)