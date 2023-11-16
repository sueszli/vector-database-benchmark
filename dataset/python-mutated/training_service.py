from __future__ import annotations
__all__ = ['TrainingServiceExecutionEngine']
import logging
import sys
import time
import weakref
from threading import Event, Thread
from typing import Iterable, TYPE_CHECKING, Any, cast
import nni
from nni.runtime.tuner_command_channel import command_type, TunerCommandChannel
from nni.typehint import TrialMetric
from nni.utils import MetricType
from nni.nas.space import ExecutableModelSpace, ModelStatus, GraphModelSpace
from .engine import ExecutionEngine
from .event import FinalMetricEvent, IntermediateMetricEvent, TrainingEndEvent
if TYPE_CHECKING:
    from nni.nas.experiment import NasExperiment
_logger = logging.getLogger(__name__)

class TrainingServiceExecutionEngine(ExecutionEngine):
    """
    The execution engine will submit every model onto training service.

    Resource management is implemented in this class.

    This engine doesn't include any optimization across graphs.

    NOTE: Due to the design of `nni.experiment`,
    the execution engine resorts to NasExperiment to submit trials as well as waiting for results.
    This is not ideal, because this engine might be one of the very few engines which need the training service.
    Ideally, the training service should be a part of the execution engine, not the experiment.

    Ideally, this class should not have any states. Its save and load methods should be empty.

    Parameters
    ----------
    nodejs_binding
        The nodejs binding of the experiment.
    fetch_intermediates
        Whether to fetch intermediate results from the training service when list models.
        Setting it to false for large-scale experiments can improve performance.
    """

    def __init__(self, nodejs_binding: NasExperiment, fetch_intermediates: bool=True) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.nodejs_binding = nodejs_binding
        self.fetch_intermediates = fetch_intermediates
        self._models: dict[int, weakref.ReferenceType[ExecutableModelSpace]] = dict()
        self._submitted_cache: dict[int, ExecutableModelSpace] = dict()
        self._current_parameter_id: int | None = None
        self._workers = 0
        self._channel = TunerCommandChannel(nodejs_binding.tuner_command_channel)
        self._channel.on_initialize(self._initialize_callback)
        self._channel.on_request_trial_jobs(self._request_trial_jobs_callback)
        self._channel.on_report_metric_data(self._report_metric_data_callback)
        self._channel.on_trial_end(self._trial_end_callback)
        self._channel.connect()
        self._channel_listen_stop_event = Event()
        self._channel_listen_thread = Thread(target=self._channel.listen, kwargs={'stop_event': self._channel_listen_stop_event}, daemon=True)
        self._channel_listen_thread.start()
        self._stopped = False

    def wait_models(self, *models: ExecutableModelSpace) -> None:
        if False:
            return 10
        "Wait models to finish training.\n\n        If argument models is empty, wait for all models to finish.\n        Using the experiment status as an indicator of all models' status,\n        which is more efficient.\n\n        For the models to receive status changes, the models must be the exact same instances as the ones submitted.\n        Dumping and reloading the models, or retrieving the unsaved models from :meth:`list_models` won't work.\n        "
        if not models:
            self._check_running()
            _logger.debug("Waiting for models. Using experiment status as an indicator of all models' status.")
            training_model_patience = 0
            while True:
                status = self.nodejs_binding.get_status()
                if status in ['DONE', 'STOPPED', 'ERROR']:
                    return
                stats = self.nodejs_binding.get_job_statistics()
                training_models_found = False
                for stat in stats:
                    if self._interpret_trial_job_status(stat['trialJobStatus']) == ModelStatus.Training:
                        training_models_found = True
                        break
                if training_models_found:
                    if training_model_patience != 0:
                        _logger.debug('Running models found. Resetting patience. Current stats: %s', stats)
                        training_model_patience = 0
                else:
                    _logger.debug('Waiting for running models to show up (patience: %d). Current stats: %s', training_model_patience, stats)
                    training_model_patience += 1
                    if training_model_patience > 6:
                        _logger.debug('No running models found. Assuming all models are done.')
                        return
                time.sleep(1)
        super().wait_models(*models)

    def submit_models(self, *models: ExecutableModelSpace) -> None:
        if False:
            i = 10
            return i + 15
        'Submit models to training service.\n\n        See Also\n        --------\n        nni.nas.ExecutionEngine.submit_models\n        '
        self._check_running()
        for model in models:
            if self._workers <= 0:
                _logger.debug('Submitted models exceed concurrency. Remaining concurrency is %d.', self._workers)
            parameter_id = self._next_parameter_id()
            self._models[parameter_id] = weakref.ref(model)
            self._submitted_cache[parameter_id] = model
            placement = None
            if isinstance(model, GraphModelSpace):
                placement = model.export_placement_constraint()
            self._channel.send_trial(parameter_id=parameter_id, parameters=cast(Any, model), placement_constraint=placement)
            model.status = ModelStatus.Training
            self._workers -= 1
            _logger.debug('Submitted model with parameter id %d. Remaining resource: %d.', parameter_id, self._workers)

    def list_models(self, status: ModelStatus | None=None) -> Iterable[ExecutableModelSpace]:
        if False:
            return 10
        'Retrieve models previously submitted.\n\n        To support a large-scale experiments with thousands of trials,\n        this method will retrieve the models from the nodejs binding (i.e., from the database).\n        The model instances will be re-created on the fly based on the data from database.\n        Although they are the same models semantically, they might not be the same instances.\n        Exceptions are those still used by the strategy.\n        Their weak references are kept in the engine and thus the exact same instances are returned.\n\n        Parameters\n        ----------\n        status\n            The status of the models to be retrieved.\n            If None, all models will be retrieved.\n        include_intermediates\n            Whether to include intermediate models.\n        '
        self._check_running()
        for trial in self.nodejs_binding.list_trial_jobs():
            if len(trial.hyperParameters) != 1:
                _logger.warning('Found trial "%s" with unexpected number of parameters. It may not be submitted by the engine. Skip.', trial.trialJobId)
                continue
            param = trial.hyperParameters[0]
            parameter_id = param.parameter_id
            model = self._find_reference_model(parameter_id)
            if model is not None:
                model_status = model.status
            else:
                model_status = self._interpret_trial_job_status(trial.status)
            if status is not None and model_status != status:
                continue
            if model is None:
                model: ExecutableModelSpace = nni.load(nni.dump(param.parameters))
                if not isinstance(model, ExecutableModelSpace):
                    _logger.error('The parameter of trial "%s" is not a model. Skip.', trial.trialJobId)
                    continue
                model.status = model_status
                if trial.finalMetricData:
                    if len(trial.finalMetricData) != 1:
                        _logger.warning('The final metric data of trial "%s" is not a single value. Taking the last one.', trial.trialJobId)
                    model.metrics.final = cast(TrialMetric, trial.finalMetricData[-1].data)
                if self.fetch_intermediates:
                    metrics = self.nodejs_binding.get_job_metrics(trial.trialJobId)
                    for metric_data in metrics.get(trial.trialJobId, []):
                        if metric_data.type == 'PERIODICAL':
                            model.metrics.add_intermediate(metric_data.data)
            yield model
        for model in self._submitted_cache.values():
            if status is None or model.status == status:
                yield model

    def idle_worker_available(self) -> bool:
        if False:
            while True:
                i = 10
        'Return the number of available resources.\n\n        The resource is maintained by the engine itself.\n        It should be fetched from nodejs side directly in future.\n        '
        return self._workers > 0

    def budget_available(self) -> bool:
        if False:
            return 10
        'Infer the budget from resources.\n\n        This should have a dedicated implementation on the nodejs side in the future.\n        '
        self._check_running()
        return self.nodejs_binding.get_status() in ['INITIALIZED', 'RUNNING', 'TUNER_NO_MORE_TRIAL']

    def shutdown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._stopped = True
        self._channel_listen_stop_event.set()
        self._channel_listen_thread.join()

    def load_state_dict(self, state_dict: dict) -> None:
        if False:
            print('Hello World!')
        _logger.info('Loading state for training service engine does nothing.')

    def state_dict(self) -> dict:
        if False:
            print('Hello World!')
        return {}

    def _initialize_callback(self, command: command_type.Initialize) -> None:
        if False:
            while True:
                i = 10
        self._channel.send_initialized()

    def _request_trial_jobs_callback(self, command: command_type.RequestTrialJobs) -> None:
        if False:
            print('Hello World!')
        self._workers += command.count
        _logger.debug('New resources received. Remaining resource: %d.', self._workers)

    def _report_metric_data_callback(self, command: command_type.ReportMetricData) -> None:
        if False:
            while True:
                i = 10
        model = self._find_reference_model(command.parameter_id)
        if model is not None:
            if command.type == MetricType.PERIODICAL:
                self.dispatch_model_event(IntermediateMetricEvent(model, cast(TrialMetric, command.value)))
            elif command.type == MetricType.FINAL:
                self.dispatch_model_event(FinalMetricEvent(model, cast(TrialMetric, command.value)))
            else:
                raise ValueError('Unknown metric type: %r' % command.type)
        else:
            _logger.debug('Received metric data of "%s" (parameter id: %d) but the model has been garbage-collected. Skip.', command.trial_job_id, command.parameter_id)

    def _trial_end_callback(self, command: command_type.TrialEnd) -> None:
        if False:
            for i in range(10):
                print('nop')
        if len(command.parameter_ids) != 1:
            _logger.warning('Received trial end event of "%s" with unexpected number of parameters. It may not be submitted by the engine. Skip.', command.trial_job_id)
        else:
            model = self._find_reference_model(command.parameter_ids[0])
            if model is not None:
                model_status = self._interpret_trial_job_status(command.event)
                self.dispatch_model_event(TrainingEndEvent(model, model_status))
            else:
                _logger.debug('Received trial end event of "%s" (parameter id: %d) but the model has been garbage-collected. Skip.', command.trial_job_id, command.parameter_ids[0])

    def _check_running(self) -> None:
        if False:
            while True:
                i = 10
        if self._stopped:
            raise RuntimeError('The engine has been stopped. Cannot take any more action.')

    def _next_parameter_id(self) -> int:
        if False:
            i = 10
            return i + 15
        'Get the next available parameter id.\n\n        Communicate with nodejs binding if necessary.\n        '
        if self._current_parameter_id is None:
            trials = self.nodejs_binding.list_trial_jobs()
            existing_ids = [param.parameter_id for trial in trials for param in trial.hyperParameters]
            self._current_parameter_id = max(existing_ids) if existing_ids else -1
        self._current_parameter_id += 1
        return self._current_parameter_id

    def _find_reference_model(self, parameter_id: int) -> ExecutableModelSpace | None:
        if False:
            print('Hello World!')
        'Retrieve the reference model by a parameter id.\n\n        The reference model is the model instance submitted by the strategy.\n        It is used to create a new model instance based on the information provided by the nodejs binding.\n        '
        self._invalidate_submitted_cache(parameter_id)
        if parameter_id in self._models:
            model = self._models[parameter_id]()
            if model is not None:
                return model
            _logger.debug('The reference model for parameter "%d" has been garbage-collected. Removing it from cache.', parameter_id)
            self._models.pop(parameter_id)
        return None

    def _invalidate_submitted_cache(self, parameter_id: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Remove the cache item when the parameter id has been found in the database of NNI manager.'
        self._submitted_cache.pop(parameter_id, None)

    def _interpret_trial_job_status(self, status: str) -> ModelStatus:
        if False:
            print('Hello World!')
        'Translate the trial job status into a model status.'
        if status in ['WAITING', 'RUNNING', 'UNKNOWN']:
            return ModelStatus.Training
        if status == 'SUCCEEDED':
            return ModelStatus.Trained
        return ModelStatus.Failed

def trial_entry() -> None:
    if False:
        i = 10
        return i + 15
    'The entry point for the trial job launched by training service.'
    params = nni.get_next_parameter()
    assert isinstance(params, ExecutableModelSpace), 'Generated parameter should be an ExecutableModelSpace.'
    params.execute()
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: python -m nni.nas.execution.training_service trial', file=sys.stderr)
        sys.exit(1)
    if sys.argv[1] == 'trial':
        trial_entry()