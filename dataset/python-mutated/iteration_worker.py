import abc
import logging
from typing import Final, Optional
from localstack.aws.api.stepfunctions import HistoryEventType, MapIterationEventDetails
from localstack.services.stepfunctions.asl.component.common.error_name.custom_error_name import CustomErrorName
from localstack.services.stepfunctions.asl.component.common.error_name.failure_event import FailureEvent, FailureEventException
from localstack.services.stepfunctions.asl.component.state.state_execution.state_map.iteration.job import Job, JobPool
from localstack.services.stepfunctions.asl.eval.contextobject.contex_object import Item, Map
from localstack.services.stepfunctions.asl.eval.environment import Environment
from localstack.services.stepfunctions.asl.eval.event.event_detail import EventDetails
from localstack.services.stepfunctions.asl.eval.program_state import ProgramError, ProgramState, ProgramStopped
LOG = logging.getLogger(__name__)

class IterationWorker(abc.ABC):
    _work_name: Final[str]
    _job_pool: Final[JobPool]
    _env: Final[Environment]
    _stop_signal_received: bool

    def __init__(self, work_name: str, job_pool: JobPool, env: Environment):
        if False:
            for i in range(10):
                print('nop')
        self._work_name = work_name
        self._job_pool = job_pool
        self._env = env
        self._stop_signal_received = False

    def sig_stop(self):
        if False:
            while True:
                i = 10
        self._stop_signal_received = True

    def stopped(self):
        if False:
            return 10
        return self._stop_signal_received

    @abc.abstractmethod
    def _eval_input(self, env_frame: Environment) -> None:
        if False:
            i = 10
            return i + 15
        ...

    def _eval_job(self, env: Environment, job: Job) -> None:
        if False:
            i = 10
            return i + 15
        map_iteration_event_details = MapIterationEventDetails(name=self._work_name, index=job.job_index)
        env.event_history.add_event(context=env.event_history_context, hist_type_event=HistoryEventType.MapIterationStarted, event_detail=EventDetails(mapIterationStartedEventDetails=map_iteration_event_details))
        job_output = RuntimeError(f"Unexpected Runtime Error in ItemProcessor worker for input '{job.job_index}'.")
        try:
            env.context_object_manager.context_object['Map'] = Map(Item=Item(Index=job.job_index, Value=job.job_input))
            env.inp = job.job_input
            self._eval_input(env_frame=env)
            job.job_program.eval(env)
            end_program_state: ProgramState = env.program_state()
            if isinstance(end_program_state, ProgramError):
                error_name = end_program_state.error.get('error')
                if error_name is not None:
                    error_name = CustomErrorName(error_name=error_name)
                raise FailureEventException(failure_event=FailureEvent(error_name=error_name, event_type=HistoryEventType.MapIterationFailed, event_details=EventDetails(executionFailedEventDetails=end_program_state.error)))
            elif isinstance(end_program_state, ProgramStopped):
                raise FailureEventException(failure_event=FailureEvent(error_name=CustomErrorName(error_name=HistoryEventType.MapIterationAborted), event_type=HistoryEventType.MapIterationAborted, event_details=EventDetails(executionFailedEventDetails=end_program_state.error)))
            env.event_history.add_event(context=env.event_history_context, hist_type_event=HistoryEventType.MapIterationSucceeded, event_detail=EventDetails(mapIterationSucceededEventDetails=map_iteration_event_details), update_source_event_id=False)
            job_output = env.inp
        except FailureEventException as failure_event_ex:
            job_output = failure_event_ex
            if failure_event_ex.failure_event.event_type == HistoryEventType.MapIterationAborted:
                env.event_history.add_event(context=env.event_history_context, hist_type_event=HistoryEventType.MapIterationAborted, event_detail=EventDetails(mapIterationAbortedEventDetails=map_iteration_event_details), update_source_event_id=False)
            else:
                env.event_history.add_event(context=env.event_history_context, hist_type_event=HistoryEventType.MapIterationFailed, event_detail=EventDetails(mapIterationFailedEventDetails=map_iteration_event_details), update_source_event_id=False)
        except Exception as ex:
            LOG.warning(f"Unhandled termination error in item processor worker for job '{job.job_index}'.")
            job_output = ex
            env.event_history.add_event(context=env.event_history_context, hist_type_event=HistoryEventType.MapIterationFailed, event_detail=EventDetails(mapIterationFailedEventDetails=map_iteration_event_details), update_source_event_id=False)
        finally:
            job.job_output = job_output

    def _eval_pool(self, job: Optional[Job], worker_frame: Environment) -> None:
        if False:
            print('Hello World!')
        if job is None:
            self._env.close_frame(worker_frame)
            return
        job_frame = worker_frame.open_frame()
        self._eval_job(env=job_frame, job=job)
        worker_frame.close_frame(job_frame)
        if isinstance(job.job_output, Exception):
            self._env.close_frame(worker_frame)
            self._job_pool.close_job(job)
            return
        if self.stopped():
            self._env.close_frame(worker_frame)
            self._job_pool.close_job(job)
            return
        next_job: Job = self._job_pool.next_job()
        if next_job is None:
            worker_frame.event_history_context.source_event_id = job_frame.event_history_context.last_published_event_id
            self._env.close_frame(worker_frame)
            self._job_pool.close_job(job)
            return
        self._job_pool.close_job(job)
        self._eval_pool(job=next_job, worker_frame=worker_frame)

    def eval(self) -> None:
        if False:
            i = 10
            return i + 15
        self._eval_pool(job=self._job_pool.next_job(), worker_frame=self._env.open_frame())