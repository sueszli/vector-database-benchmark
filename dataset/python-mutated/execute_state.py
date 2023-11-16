import abc
import copy
import logging
import threading
from threading import Thread
from typing import Any, Optional
from localstack.aws.api.stepfunctions import HistoryEventType, TaskFailedEventDetails
from localstack.services.stepfunctions.asl.component.common.catch.catch_decl import CatchDecl
from localstack.services.stepfunctions.asl.component.common.catch.catch_outcome import CatchOutcome
from localstack.services.stepfunctions.asl.component.common.error_name.failure_event import FailureEvent, FailureEventException
from localstack.services.stepfunctions.asl.component.common.error_name.states_error_name import StatesErrorName
from localstack.services.stepfunctions.asl.component.common.error_name.states_error_name_type import StatesErrorNameType
from localstack.services.stepfunctions.asl.component.common.path.result_path import ResultPath
from localstack.services.stepfunctions.asl.component.common.result_selector import ResultSelector
from localstack.services.stepfunctions.asl.component.common.retry.retry_decl import RetryDecl
from localstack.services.stepfunctions.asl.component.common.retry.retry_outcome import RetryOutcome
from localstack.services.stepfunctions.asl.component.common.timeouts.heartbeat import Heartbeat, HeartbeatSeconds
from localstack.services.stepfunctions.asl.component.common.timeouts.timeout import EvalTimeoutError, Timeout, TimeoutSeconds
from localstack.services.stepfunctions.asl.component.state.state import CommonStateField
from localstack.services.stepfunctions.asl.component.state.state_props import StateProps
from localstack.services.stepfunctions.asl.eval.environment import Environment
from localstack.services.stepfunctions.asl.eval.event.event_detail import EventDetails
LOG = logging.getLogger(__name__)

class ExecutionState(CommonStateField, abc.ABC):

    def __init__(self, state_entered_event_type: HistoryEventType, state_exited_event_type: Optional[HistoryEventType]):
        if False:
            while True:
                i = 10
        super().__init__(state_entered_event_type=state_entered_event_type, state_exited_event_type=state_exited_event_type)
        self.result_path: Optional[ResultPath] = None
        self.result_selector: Optional[ResultSelector] = None
        self.retry: Optional[RetryDecl] = None
        self.catch: Optional[CatchDecl] = None
        self.timeout: Timeout = TimeoutSeconds(timeout_seconds=TimeoutSeconds.DEFAULT_TIMEOUT_SECONDS)
        self.heartbeat: Optional[Heartbeat] = None

    def from_state_props(self, state_props: StateProps) -> None:
        if False:
            i = 10
            return i + 15
        super().from_state_props(state_props=state_props)
        self.result_path = state_props.get(ResultPath)
        self.result_selector = state_props.get(ResultSelector)
        self.retry = state_props.get(RetryDecl)
        self.catch = state_props.get(CatchDecl)
        timeout = state_props.get(Timeout)
        heartbeat = state_props.get(Heartbeat)
        if isinstance(timeout, TimeoutSeconds) and isinstance(heartbeat, HeartbeatSeconds):
            if timeout.timeout_seconds <= heartbeat.heartbeat_seconds:
                raise RuntimeError(f"'HeartbeatSeconds' interval MUST be smaller than the 'TimeoutSeconds' value, got '{timeout.timeout_seconds}' and '{heartbeat.heartbeat_seconds}' respectively.")
        if heartbeat is not None and timeout is None:
            timeout = TimeoutSeconds(timeout_seconds=60, is_default=True)
        if timeout is not None:
            self.timeout = timeout
        if heartbeat is not None:
            self.heartbeat = heartbeat

    def _from_error(self, env: Environment, ex: Exception) -> FailureEvent:
        if False:
            while True:
                i = 10
        if isinstance(ex, FailureEventException):
            return ex.failure_event
        LOG.warning('State Task encountered an unhandled exception that lead to a State.Runtime error.')
        return FailureEvent(error_name=StatesErrorName(typ=StatesErrorNameType.StatesRuntime), event_type=HistoryEventType.TaskFailed, event_details=EventDetails(taskFailedEventDetails=TaskFailedEventDetails(error=StatesErrorNameType.StatesRuntime.to_name(), cause=str(ex))))

    @abc.abstractmethod
    def _eval_execution(self, env: Environment) -> None:
        if False:
            while True:
                i = 10
        ...

    def _handle_retry(self, env: Environment, failure_event: FailureEvent) -> RetryOutcome:
        if False:
            for i in range(10):
                print('nop')
        env.stack.append(failure_event.error_name)
        self.retry.eval(env)
        res: RetryOutcome = env.stack.pop()
        if res == RetryOutcome.CanRetry:
            retry_count = env.context_object_manager.context_object['State']['RetryCount']
            env.context_object_manager.context_object['State']['RetryCount'] = retry_count + 1
        return res

    def _handle_catch(self, env: Environment, failure_event: FailureEvent) -> CatchOutcome:
        if False:
            for i in range(10):
                print('nop')
        env.stack.append(failure_event)
        self.catch.eval(env)
        res: CatchOutcome = env.stack.pop()
        return res

    def _handle_uncaught(self, env: Environment, failure_event: FailureEvent) -> None:
        if False:
            return 10
        self._terminate_with_event(env=env, failure_event=failure_event)

    @staticmethod
    def _terminate_with_event(env: Environment, failure_event: FailureEvent) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise FailureEventException(failure_event=failure_event)

    def _evaluate_with_timeout(self, env: Environment) -> None:
        if False:
            return 10
        self.timeout.eval(env=env)
        timeout_seconds: int = env.stack.pop()
        frame: Environment = env.open_frame()
        frame.inp = copy.deepcopy(env.inp)
        frame.stack = copy.deepcopy(env.stack)
        execution_outputs: list[Any] = list()
        execution_exceptions: list[Optional[Exception]] = [None]
        terminated_event = threading.Event()

        def _exec_and_notify():
            if False:
                print('Hello World!')
            try:
                self._eval_execution(frame)
                execution_outputs.extend(frame.stack)
            except Exception as ex:
                execution_exceptions.append(ex)
            terminated_event.set()
        thread = Thread(target=_exec_and_notify)
        thread.start()
        finished_on_time: bool = terminated_event.wait(timeout_seconds)
        frame.set_ended()
        env.close_frame(frame)
        execution_exception = execution_exceptions.pop()
        if execution_exception:
            raise execution_exception
        if not finished_on_time:
            raise EvalTimeoutError()
        execution_output = execution_outputs.pop()
        env.stack.append(execution_output)
        if self.result_selector:
            self.result_selector.eval(env=env)
        if self.result_path:
            self.result_path.eval(env)
        else:
            res = env.stack.pop()
            env.inp = res

    def _eval_state(self, env: Environment) -> None:
        if False:
            i = 10
            return i + 15
        env.context_object_manager.context_object['State']['RetryCount'] = 0
        while True:
            try:
                self._evaluate_with_timeout(env)
                break
            except Exception as ex:
                failure_event: FailureEvent = self._from_error(env=env, ex=ex)
                env.event_history.add_event(context=env.event_history_context, hist_type_event=failure_event.event_type, event_detail=failure_event.event_details)
                if self.retry:
                    retry_outcome: RetryOutcome = self._handle_retry(env=env, failure_event=failure_event)
                    if retry_outcome == RetryOutcome.CanRetry:
                        continue
                if self.catch:
                    catch_outcome: CatchOutcome = self._handle_catch(env=env, failure_event=failure_event)
                    if catch_outcome == CatchOutcome.Caught:
                        break
                self._handle_uncaught(env=env, failure_event=failure_event)