from __future__ import annotations
import datetime
from traceback import format_exception
from typing import TYPE_CHECKING, Any, Iterable
from sqlalchemy import Column, Integer, String, delete, func, or_, select, update
from sqlalchemy.orm import joinedload, relationship
from sqlalchemy.sql.functions import coalesce
from airflow.api_internal.internal_api_call import internal_api_call
from airflow.models.base import Base
from airflow.models.taskinstance import TaskInstance
from airflow.utils import timezone
from airflow.utils.retries import run_with_db_retries
from airflow.utils.session import NEW_SESSION, provide_session
from airflow.utils.sqlalchemy import ExtendedJSON, UtcDateTime, with_row_locks
from airflow.utils.state import TaskInstanceState
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from airflow.triggers.base import BaseTrigger

class Trigger(Base):
    """
    Base Trigger class.

    Triggers are a workload that run in an asynchronous event loop shared with
    other Triggers, and fire off events that will unpause deferred Tasks,
    start linked DAGs, etc.

    They are persisted into the database and then re-hydrated into a
    "triggerer" process, where many are run at once. We model it so that
    there is a many-to-one relationship between Task and Trigger, for future
    deduplication logic to use.

    Rows will be evicted from the database when the triggerer detects no
    active Tasks/DAGs using them. Events are not stored in the database;
    when an Event is fired, the triggerer will directly push its data to the
    appropriate Task/DAG.
    """
    __tablename__ = 'trigger'
    id = Column(Integer, primary_key=True)
    classpath = Column(String(1000), nullable=False)
    kwargs = Column(ExtendedJSON, nullable=False)
    created_date = Column(UtcDateTime, nullable=False)
    triggerer_id = Column(Integer, nullable=True)
    triggerer_job = relationship('Job', primaryjoin='Job.id == Trigger.triggerer_id', foreign_keys=triggerer_id, uselist=False)
    task_instance = relationship('TaskInstance', back_populates='trigger', lazy='joined', uselist=False)

    def __init__(self, classpath: str, kwargs: dict[str, Any], created_date: datetime.datetime | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.classpath = classpath
        self.kwargs = kwargs
        self.created_date = created_date or timezone.utcnow()

    @classmethod
    @internal_api_call
    def from_object(cls, trigger: BaseTrigger) -> Trigger:
        if False:
            print('Hello World!')
        'Alternative constructor that creates a trigger row based directly off of a Trigger object.'
        (classpath, kwargs) = trigger.serialize()
        return cls(classpath=classpath, kwargs=kwargs)

    @classmethod
    @internal_api_call
    @provide_session
    def bulk_fetch(cls, ids: Iterable[int], session: Session=NEW_SESSION) -> dict[int, Trigger]:
        if False:
            while True:
                i = 10
        'Fetch all the Triggers by ID and return a dict mapping ID -> Trigger instance.'
        query = session.scalars(select(cls).where(cls.id.in_(ids)).options(joinedload('task_instance'), joinedload('task_instance.trigger'), joinedload('task_instance.trigger.triggerer_job')))
        return {obj.id: obj for obj in query}

    @classmethod
    @internal_api_call
    @provide_session
    def clean_unused(cls, session: Session=NEW_SESSION) -> None:
        if False:
            i = 10
            return i + 15
        'Delete all triggers that have no tasks dependent on them.\n\n        Triggers have a one-to-many relationship to task instances, so we need\n        to clean those up first. Afterwards we can drop the triggers not\n        referenced by anyone.\n        '
        for attempt in run_with_db_retries():
            with attempt:
                session.execute(update(TaskInstance).where(TaskInstance.state != TaskInstanceState.DEFERRED, TaskInstance.trigger_id.is_not(None)).values(trigger_id=None))
        ids = session.scalars(select(cls.id).join(TaskInstance, cls.id == TaskInstance.trigger_id, isouter=True).group_by(cls.id).having(func.count(TaskInstance.trigger_id) == 0)).all()
        session.execute(delete(Trigger).where(Trigger.id.in_(ids)).execution_options(synchronize_session=False))

    @classmethod
    @internal_api_call
    @provide_session
    def submit_event(cls, trigger_id, event, session: Session=NEW_SESSION) -> None:
        if False:
            i = 10
            return i + 15
        'Take an event from an instance of itself, and trigger all dependent tasks to resume.'
        for task_instance in session.scalars(select(TaskInstance).where(TaskInstance.trigger_id == trigger_id, TaskInstance.state == TaskInstanceState.DEFERRED)):
            next_kwargs = task_instance.next_kwargs or {}
            next_kwargs['event'] = event.payload
            task_instance.next_kwargs = next_kwargs
            task_instance.trigger_id = None
            task_instance.state = TaskInstanceState.SCHEDULED

    @classmethod
    @internal_api_call
    @provide_session
    def submit_failure(cls, trigger_id, exc=None, session: Session=NEW_SESSION) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        When a trigger has failed unexpectedly, mark everything that depended on it as failed.\n\n        Notably, we have to actually run the failure code from a worker as it may\n        have linked callbacks, so hilariously we have to re-schedule the task\n        instances to a worker just so they can then fail.\n\n        We use a special __fail__ value for next_method to achieve this that\n        the runtime code understands as immediate-fail, and pack the error into\n        next_kwargs.\n\n        TODO: Once we have shifted callback (and email) handling to run on\n        workers as first-class concepts, we can run the failure code here\n        in-process, but we can't do that right now.\n        "
        for task_instance in session.scalars(select(TaskInstance).where(TaskInstance.trigger_id == trigger_id, TaskInstance.state == TaskInstanceState.DEFERRED)):
            traceback = format_exception(type(exc), exc, exc.__traceback__) if exc else None
            task_instance.next_method = '__fail__'
            task_instance.next_kwargs = {'error': 'Trigger failure', 'traceback': traceback}
            task_instance.trigger_id = None
            task_instance.state = TaskInstanceState.SCHEDULED

    @classmethod
    @internal_api_call
    @provide_session
    def ids_for_triggerer(cls, triggerer_id, session: Session=NEW_SESSION) -> list[int]:
        if False:
            return 10
        'Retrieve a list of triggerer_ids.'
        return session.scalars(select(cls.id).where(cls.triggerer_id == triggerer_id)).all()

    @classmethod
    @internal_api_call
    @provide_session
    def assign_unassigned(cls, triggerer_id, capacity, health_check_threshold, session: Session=NEW_SESSION) -> None:
        if False:
            print('Hello World!')
        '\n        Assign unassigned triggers based on a number of conditions.\n\n        Takes a triggerer_id, the capacity for that triggerer and the Triggerer job heartrate\n        health check threshold, and assigns unassigned triggers until that capacity is reached,\n        or there are no more unassigned triggers.\n        '
        from airflow.jobs.job import Job
        count = session.scalar(select(func.count(cls.id)).filter(cls.triggerer_id == triggerer_id))
        capacity -= count
        if capacity <= 0:
            return
        alive_triggerer_ids = session.scalars(select(Job.id).where(Job.end_date.is_(None), Job.latest_heartbeat > timezone.utcnow() - datetime.timedelta(seconds=health_check_threshold), Job.job_type == 'TriggererJob')).all()
        trigger_ids_query = cls.get_sorted_triggers(capacity=capacity, alive_triggerer_ids=alive_triggerer_ids, session=session)
        if trigger_ids_query:
            session.execute(update(cls).where(cls.id.in_([i.id for i in trigger_ids_query])).values(triggerer_id=triggerer_id).execution_options(synchronize_session=False))
        session.commit()

    @classmethod
    def get_sorted_triggers(cls, capacity, alive_triggerer_ids, session):
        if False:
            i = 10
            return i + 15
        query = with_row_locks(select(cls.id).join(TaskInstance, cls.id == TaskInstance.trigger_id, isouter=False).where(or_(cls.triggerer_id.is_(None), cls.triggerer_id.not_in(alive_triggerer_ids))).order_by(coalesce(TaskInstance.priority_weight, 0).desc(), cls.created_date).limit(capacity), session, skip_locked=True)
        return session.execute(query).all()