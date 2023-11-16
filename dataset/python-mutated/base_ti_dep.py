from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple
from airflow.ti_deps.dep_context import DepContext
from airflow.utils.session import provide_session
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from airflow.models.taskinstance import TaskInstance

class BaseTIDep:
    """
    Abstract base class for task instances dependencies.

    All dependencies must be satisfied in order for task instances to run.
    For example, a task that can only run if a certain number of its upstream tasks succeed.
    This is an abstract class and must be subclassed to be used.
    """
    IGNORABLE = False
    IS_TASK_DEP = False

    def __eq__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return isinstance(self, type(other))

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return hash(type(self))

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<TIDep({self.name})>'

    @property
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        'The human-readable name for the dependency.\n\n        Use the class name as the default if ``NAME`` is not provided.\n        '
        return getattr(self, 'NAME', self.__class__.__name__)

    def _get_dep_statuses(self, ti: TaskInstance, session: Session, dep_context: DepContext) -> Iterator[TIDepStatus]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Abstract method that returns an iterable of TIDepStatus objects.\n\n        Each object describes whether the given task instance has this dependency met.\n\n        For example a subclass could return an iterable of TIDepStatus objects, each one\n        representing if each of the passed in task's upstream tasks succeeded or not.\n\n        :param ti: the task instance to get the dependency status for\n        :param session: database session\n        :param dep_context: the context for which this dependency should be evaluated for\n        "
        raise NotImplementedError

    @provide_session
    def get_dep_statuses(self, ti: TaskInstance, session: Session, dep_context: DepContext | None=None) -> Iterator[TIDepStatus]:
        if False:
            print('Hello World!')
        '\n        Wrap around the private _get_dep_statuses method.\n\n        Contains some global checks for all dependencies.\n\n        :param ti: the task instance to get the dependency status for\n        :param session: database session\n        :param dep_context: the context for which this dependency should be evaluated for\n        '
        cxt = DepContext() if dep_context is None else dep_context
        if self.IGNORABLE and cxt.ignore_all_deps:
            yield self._passing_status(reason='Context specified all dependencies should be ignored.')
            return
        if self.IS_TASK_DEP and cxt.ignore_task_deps:
            yield self._passing_status(reason='Context specified all task dependencies should be ignored.')
            return
        yield from self._get_dep_statuses(ti, session, cxt)

    @provide_session
    def is_met(self, ti: TaskInstance, session: Session, dep_context: DepContext | None=None) -> bool:
        if False:
            print('Hello World!')
        '\n        Return whether a dependency is met for a given task instance.\n\n        A dependency is considered met if all the dependency statuses it reports are passing.\n\n        :param ti: the task instance to see if this dependency is met for\n        :param session: database session\n        :param dep_context: The context this dependency is being checked under that stores\n            state that can be used by this dependency.\n        '
        return all((status.passed for status in self.get_dep_statuses(ti, session, dep_context)))

    @provide_session
    def get_failure_reasons(self, ti: TaskInstance, session: Session, dep_context: DepContext | None=None) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        "\n        Return an iterable of strings that explain why this dependency wasn't met.\n\n        :param ti: the task instance to see if this dependency is met for\n        :param session: database session\n        :param dep_context: The context this dependency is being checked under that stores\n            state that can be used by this dependency.\n        "
        for dep_status in self.get_dep_statuses(ti, session, dep_context):
            if not dep_status.passed:
                yield dep_status.reason

    def _failing_status(self, reason: str='') -> TIDepStatus:
        if False:
            print('Hello World!')
        return TIDepStatus(self.name, False, reason)

    def _passing_status(self, reason: str='') -> TIDepStatus:
        if False:
            return 10
        return TIDepStatus(self.name, True, reason)

class TIDepStatus(NamedTuple):
    """Dependency status for a task instance indicating whether the task instance passed the dependency."""
    dep_name: str
    passed: bool
    reason: str