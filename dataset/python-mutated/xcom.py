from __future__ import annotations
import collections.abc
import contextlib
import inspect
import itertools
import json
import logging
import pickle
import warnings
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Any, Generator, Iterable, cast, overload
import attr
from sqlalchemy import Column, ForeignKeyConstraint, Index, Integer, LargeBinary, PrimaryKeyConstraint, String, delete, text
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import Query, reconstructor, relationship
from sqlalchemy.orm.exc import NoResultFound
from airflow import settings
from airflow.api_internal.internal_api_call import internal_api_call
from airflow.configuration import conf
from airflow.exceptions import RemovedInAirflow3Warning
from airflow.models.base import COLLATION_ARGS, ID_LEN, Base
from airflow.utils import timezone
from airflow.utils.helpers import exactly_one, is_container
from airflow.utils.json import XComDecoder, XComEncoder
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.session import NEW_SESSION, provide_session
from airflow.utils.sqlalchemy import UtcDateTime
from airflow.utils.xcom import MAX_XCOM_SIZE, XCOM_RETURN_KEY
log = logging.getLogger(__name__)
if TYPE_CHECKING:
    import datetime
    import pendulum
    from sqlalchemy.orm import Session
    from airflow.models.taskinstancekey import TaskInstanceKey

class BaseXCom(Base, LoggingMixin):
    """Base class for XCom objects."""
    __tablename__ = 'xcom'
    dag_run_id = Column(Integer(), nullable=False, primary_key=True)
    task_id = Column(String(ID_LEN, **COLLATION_ARGS), nullable=False, primary_key=True)
    map_index = Column(Integer, primary_key=True, nullable=False, server_default=text('-1'))
    key = Column(String(512, **COLLATION_ARGS), nullable=False, primary_key=True)
    dag_id = Column(String(ID_LEN, **COLLATION_ARGS), nullable=False)
    run_id = Column(String(ID_LEN, **COLLATION_ARGS), nullable=False)
    value = Column(LargeBinary)
    timestamp = Column(UtcDateTime, default=timezone.utcnow, nullable=False)
    __table_args__ = (Index('idx_xcom_key', key), Index('idx_xcom_task_instance', dag_id, task_id, run_id, map_index), PrimaryKeyConstraint('dag_run_id', 'task_id', 'map_index', 'key', name='xcom_pkey', mssql_clustered=True), ForeignKeyConstraint([dag_id, task_id, run_id, map_index], ['task_instance.dag_id', 'task_instance.task_id', 'task_instance.run_id', 'task_instance.map_index'], name='xcom_task_instance_fkey', ondelete='CASCADE'))
    dag_run = relationship('DagRun', primaryjoin='BaseXCom.dag_run_id == foreign(DagRun.id)', uselist=False, lazy='joined', passive_deletes='all')
    execution_date = association_proxy('dag_run', 'execution_date')

    @reconstructor
    def init_on_load(self):
        if False:
            print('Hello World!')
        '\n        Execute after the instance has been loaded from the DB or otherwise reconstituted; called by the ORM.\n\n        i.e automatically deserialize Xcom value when loading from DB.\n        '
        self.value = self.orm_deserialize_value()

    def __repr__(self):
        if False:
            return 10
        if self.map_index < 0:
            return f'<XCom "{self.key}" ({self.task_id} @ {self.run_id})>'
        return f'<XCom "{self.key}" ({self.task_id}[{self.map_index}] @ {self.run_id})>'

    @overload
    @classmethod
    def set(cls, key: str, value: Any, *, dag_id: str, task_id: str, run_id: str, map_index: int=-1, session: Session=NEW_SESSION) -> None:
        if False:
            print('Hello World!')
        'Store an XCom value.\n\n        A deprecated form of this function accepts ``execution_date`` instead of\n        ``run_id``. The two arguments are mutually exclusive.\n\n        :param key: Key to store the XCom.\n        :param value: XCom value to store.\n        :param dag_id: DAG ID.\n        :param task_id: Task ID.\n        :param run_id: DAG run ID for the task.\n        :param map_index: Optional map index to assign XCom for a mapped task.\n            The default is ``-1`` (set for a non-mapped task).\n        :param session: Database session. If not given, a new session will be\n            created for this function.\n        '

    @overload
    @classmethod
    def set(cls, key: str, value: Any, task_id: str, dag_id: str, execution_date: datetime.datetime, session: Session=NEW_SESSION) -> None:
        if False:
            return 10
        'Store an XCom value.\n\n        :sphinx-autoapi-skip:\n        '

    @classmethod
    @provide_session
    def set(cls, key: str, value: Any, task_id: str, dag_id: str, execution_date: datetime.datetime | None=None, session: Session=NEW_SESSION, *, run_id: str | None=None, map_index: int=-1) -> None:
        if False:
            while True:
                i = 10
        'Store an XCom value.\n\n        :sphinx-autoapi-skip:\n        '
        from airflow.models.dagrun import DagRun
        if not exactly_one(execution_date is not None, run_id is not None):
            raise ValueError(f'Exactly one of run_id or execution_date must be passed. Passed execution_date={execution_date}, run_id={run_id}')
        if run_id is None:
            message = "Passing 'execution_date' to 'XCom.set()' is deprecated. Use 'run_id' instead."
            warnings.warn(message, RemovedInAirflow3Warning, stacklevel=3)
            try:
                (dag_run_id, run_id) = session.query(DagRun.id, DagRun.run_id).filter(DagRun.dag_id == dag_id, DagRun.execution_date == execution_date).one()
            except NoResultFound:
                raise ValueError(f'DAG run not found on DAG {dag_id!r} at {execution_date}') from None
        else:
            dag_run_id = session.query(DagRun.id).filter_by(dag_id=dag_id, run_id=run_id).scalar()
            if dag_run_id is None:
                raise ValueError(f'DAG run not found on DAG {dag_id!r} with ID {run_id!r}')
        if isinstance(value, LazyXComAccess):
            warning_message = 'Coercing mapped lazy proxy %s from task %s (DAG %s, run %s) to list, which may degrade performance. Review resource requirements for this operation, and call list() to suppress this message. See Dynamic Task Mapping documentation for more information about lazy proxy objects.'
            log.warning(warning_message, 'return value' if key == XCOM_RETURN_KEY else f'value {key}', task_id, dag_id, run_id or execution_date)
            value = list(value)
        value = cls.serialize_value(value=value, key=key, task_id=task_id, dag_id=dag_id, run_id=run_id, map_index=map_index)
        session.execute(delete(cls).where(cls.key == key, cls.run_id == run_id, cls.task_id == task_id, cls.dag_id == dag_id, cls.map_index == map_index))
        new = cast(Any, cls)(dag_run_id=dag_run_id, key=key, value=value, run_id=run_id, task_id=task_id, dag_id=dag_id, map_index=map_index)
        session.add(new)
        session.flush()

    @staticmethod
    @provide_session
    @internal_api_call
    def get_value(*, ti_key: TaskInstanceKey, key: str | None=None, session: Session=NEW_SESSION) -> Any:
        if False:
            print('Hello World!')
        'Retrieve an XCom value for a task instance.\n\n        This method returns "full" XCom values (i.e. uses ``deserialize_value``\n        from the XCom backend). Use :meth:`get_many` if you want the "shortened"\n        value via ``orm_deserialize_value``.\n\n        If there are no results, *None* is returned. If multiple XCom entries\n        match the criteria, an arbitrary one is returned.\n\n        :param ti_key: The TaskInstanceKey to look up the XCom for.\n        :param key: A key for the XCom. If provided, only XCom with matching\n            keys will be returned. Pass *None* (default) to remove the filter.\n        :param session: Database session. If not given, a new session will be\n            created for this function.\n        '
        return BaseXCom.get_one(key=key, task_id=ti_key.task_id, dag_id=ti_key.dag_id, run_id=ti_key.run_id, map_index=ti_key.map_index, session=session)

    @overload
    @staticmethod
    @internal_api_call
    def get_one(*, key: str | None=None, dag_id: str | None=None, task_id: str | None=None, run_id: str | None=None, map_index: int | None=None, session: Session=NEW_SESSION) -> Any | None:
        if False:
            while True:
                i = 10
        'Retrieve an XCom value, optionally meeting certain criteria.\n\n        This method returns "full" XCom values (i.e. uses ``deserialize_value``\n        from the XCom backend). Use :meth:`get_many` if you want the "shortened"\n        value via ``orm_deserialize_value``.\n\n        If there are no results, *None* is returned. If multiple XCom entries\n        match the criteria, an arbitrary one is returned.\n\n        A deprecated form of this function accepts ``execution_date`` instead of\n        ``run_id``. The two arguments are mutually exclusive.\n\n        .. seealso:: ``get_value()`` is a convenience function if you already\n            have a structured TaskInstance or TaskInstanceKey object available.\n\n        :param run_id: DAG run ID for the task.\n        :param dag_id: Only pull XCom from this DAG. Pass *None* (default) to\n            remove the filter.\n        :param task_id: Only XCom from task with matching ID will be pulled.\n            Pass *None* (default) to remove the filter.\n        :param map_index: Only XCom from task with matching ID will be pulled.\n            Pass *None* (default) to remove the filter.\n        :param key: A key for the XCom. If provided, only XCom with matching\n            keys will be returned. Pass *None* (default) to remove the filter.\n        :param include_prior_dates: If *False* (default), only XCom from the\n            specified DAG run is returned. If *True*, the latest matching XCom is\n            returned regardless of the run it belongs to.\n        :param session: Database session. If not given, a new session will be\n            created for this function.\n        '

    @overload
    @staticmethod
    @internal_api_call
    def get_one(execution_date: datetime.datetime, key: str | None=None, task_id: str | None=None, dag_id: str | None=None, include_prior_dates: bool=False, session: Session=NEW_SESSION) -> Any | None:
        if False:
            return 10
        'Retrieve an XCom value, optionally meeting certain criteria.\n\n        :sphinx-autoapi-skip:\n        '

    @staticmethod
    @provide_session
    @internal_api_call
    def get_one(execution_date: datetime.datetime | None=None, key: str | None=None, task_id: str | None=None, dag_id: str | None=None, include_prior_dates: bool=False, session: Session=NEW_SESSION, *, run_id: str | None=None, map_index: int | None=None) -> Any | None:
        if False:
            return 10
        'Retrieve an XCom value, optionally meeting certain criteria.\n\n        :sphinx-autoapi-skip:\n        '
        if not exactly_one(execution_date is not None, run_id is not None):
            raise ValueError('Exactly one of run_id or execution_date must be passed')
        if run_id:
            query = BaseXCom.get_many(run_id=run_id, key=key, task_ids=task_id, dag_ids=dag_id, map_indexes=map_index, include_prior_dates=include_prior_dates, limit=1, session=session)
        elif execution_date is not None:
            message = "Passing 'execution_date' to 'XCom.get_one()' is deprecated. Use 'run_id' instead."
            warnings.warn(message, RemovedInAirflow3Warning, stacklevel=3)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RemovedInAirflow3Warning)
                query = BaseXCom.get_many(execution_date=execution_date, key=key, task_ids=task_id, dag_ids=dag_id, map_indexes=map_index, include_prior_dates=include_prior_dates, limit=1, session=session)
        else:
            raise RuntimeError('Should not happen?')
        result = query.with_entities(BaseXCom.value).first()
        if result:
            return BaseXCom.deserialize_value(result)
        return None

    @overload
    @staticmethod
    def get_many(*, run_id: str, key: str | None=None, task_ids: str | Iterable[str] | None=None, dag_ids: str | Iterable[str] | None=None, map_indexes: int | Iterable[int] | None=None, include_prior_dates: bool=False, limit: int | None=None, session: Session=NEW_SESSION) -> Query:
        if False:
            while True:
                i = 10
        'Composes a query to get one or more XCom entries.\n\n        This function returns an SQLAlchemy query of full XCom objects. If you\n        just want one stored value, use :meth:`get_one` instead.\n\n        A deprecated form of this function accepts ``execution_date`` instead of\n        ``run_id``. The two arguments are mutually exclusive.\n\n        :param run_id: DAG run ID for the task.\n        :param key: A key for the XComs. If provided, only XComs with matching\n            keys will be returned. Pass *None* (default) to remove the filter.\n        :param task_ids: Only XComs from task with matching IDs will be pulled.\n            Pass *None* (default) to remove the filter.\n        :param dag_ids: Only pulls XComs from specified DAGs. Pass *None*\n            (default) to remove the filter.\n        :param map_indexes: Only XComs from matching map indexes will be pulled.\n            Pass *None* (default) to remove the filter.\n        :param include_prior_dates: If *False* (default), only XComs from the\n            specified DAG run are returned. If *True*, all matching XComs are\n            returned regardless of the run it belongs to.\n        :param session: Database session. If not given, a new session will be\n            created for this function.\n        :param limit: Limiting returning XComs\n        '

    @overload
    @staticmethod
    @internal_api_call
    def get_many(execution_date: datetime.datetime, key: str | None=None, task_ids: str | Iterable[str] | None=None, dag_ids: str | Iterable[str] | None=None, map_indexes: int | Iterable[int] | None=None, include_prior_dates: bool=False, limit: int | None=None, session: Session=NEW_SESSION) -> Query:
        if False:
            while True:
                i = 10
        'Composes a query to get one or more XCom entries.\n\n        :sphinx-autoapi-skip:\n        '

    @staticmethod
    @provide_session
    @internal_api_call
    def get_many(execution_date: datetime.datetime | None=None, key: str | None=None, task_ids: str | Iterable[str] | None=None, dag_ids: str | Iterable[str] | None=None, map_indexes: int | Iterable[int] | None=None, include_prior_dates: bool=False, limit: int | None=None, session: Session=NEW_SESSION, *, run_id: str | None=None) -> Query:
        if False:
            i = 10
            return i + 15
        'Composes a query to get one or more XCom entries.\n\n        :sphinx-autoapi-skip:\n        '
        from airflow.models.dagrun import DagRun
        if not exactly_one(execution_date is not None, run_id is not None):
            raise ValueError(f'Exactly one of run_id or execution_date must be passed. Passed execution_date={execution_date}, run_id={run_id}')
        if execution_date is not None:
            message = "Passing 'execution_date' to 'XCom.get_many()' is deprecated. Use 'run_id' instead."
            warnings.warn(message, RemovedInAirflow3Warning, stacklevel=3)
        query = session.query(BaseXCom).join(BaseXCom.dag_run)
        if key:
            query = query.filter(BaseXCom.key == key)
        if is_container(task_ids):
            query = query.filter(BaseXCom.task_id.in_(task_ids))
        elif task_ids is not None:
            query = query.filter(BaseXCom.task_id == task_ids)
        if is_container(dag_ids):
            query = query.filter(BaseXCom.dag_id.in_(dag_ids))
        elif dag_ids is not None:
            query = query.filter(BaseXCom.dag_id == dag_ids)
        if isinstance(map_indexes, range) and map_indexes.step == 1:
            query = query.filter(BaseXCom.map_index >= map_indexes.start, BaseXCom.map_index < map_indexes.stop)
        elif is_container(map_indexes):
            query = query.filter(BaseXCom.map_index.in_(map_indexes))
        elif map_indexes is not None:
            query = query.filter(BaseXCom.map_index == map_indexes)
        if include_prior_dates:
            if execution_date is not None:
                query = query.filter(DagRun.execution_date <= execution_date)
            else:
                dr = session.query(DagRun.execution_date).filter(DagRun.run_id == run_id).subquery()
                query = query.filter(BaseXCom.execution_date <= dr.c.execution_date)
        elif execution_date is not None:
            query = query.filter(DagRun.execution_date == execution_date)
        else:
            query = query.filter(BaseXCom.run_id == run_id)
        query = query.order_by(DagRun.execution_date.desc(), BaseXCom.timestamp.desc())
        if limit:
            return query.limit(limit)
        return query

    @classmethod
    @provide_session
    def delete(cls, xcoms: XCom | Iterable[XCom], session: Session) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Delete one or multiple XCom entries.'
        if isinstance(xcoms, XCom):
            xcoms = [xcoms]
        for xcom in xcoms:
            if not isinstance(xcom, XCom):
                raise TypeError(f'Expected XCom; received {xcom.__class__.__name__}')
            session.delete(xcom)
        session.commit()

    @overload
    @staticmethod
    @internal_api_call
    def clear(*, dag_id: str, task_id: str, run_id: str, map_index: int | None=None, session: Session=NEW_SESSION) -> None:
        if False:
            print('Hello World!')
        'Clear all XCom data from the database for the given task instance.\n\n        A deprecated form of this function accepts ``execution_date`` instead of\n        ``run_id``. The two arguments are mutually exclusive.\n\n        :param dag_id: ID of DAG to clear the XCom for.\n        :param task_id: ID of task to clear the XCom for.\n        :param run_id: ID of DAG run to clear the XCom for.\n        :param map_index: If given, only clear XCom from this particular mapped\n            task. The default ``None`` clears *all* XComs from the task.\n        :param session: Database session. If not given, a new session will be\n            created for this function.\n        '

    @overload
    @staticmethod
    @internal_api_call
    def clear(execution_date: pendulum.DateTime, dag_id: str, task_id: str, session: Session=NEW_SESSION) -> None:
        if False:
            while True:
                i = 10
        'Clear all XCom data from the database for the given task instance.\n\n        :sphinx-autoapi-skip:\n        '

    @staticmethod
    @provide_session
    @internal_api_call
    def clear(execution_date: pendulum.DateTime | None=None, dag_id: str | None=None, task_id: str | None=None, session: Session=NEW_SESSION, *, run_id: str | None=None, map_index: int | None=None) -> None:
        if False:
            print('Hello World!')
        'Clear all XCom data from the database for the given task instance.\n\n        :sphinx-autoapi-skip:\n        '
        from airflow.models import DagRun
        if dag_id is None:
            raise TypeError('clear() missing required argument: dag_id')
        if task_id is None:
            raise TypeError('clear() missing required argument: task_id')
        if not exactly_one(execution_date is not None, run_id is not None):
            raise ValueError(f'Exactly one of run_id or execution_date must be passed. Passed execution_date={execution_date}, run_id={run_id}')
        if execution_date is not None:
            message = "Passing 'execution_date' to 'XCom.clear()' is deprecated. Use 'run_id' instead."
            warnings.warn(message, RemovedInAirflow3Warning, stacklevel=3)
            run_id = session.query(DagRun.run_id).filter(DagRun.dag_id == dag_id, DagRun.execution_date == execution_date).scalar()
        query = session.query(BaseXCom).filter_by(dag_id=dag_id, task_id=task_id, run_id=run_id)
        if map_index is not None:
            query = query.filter_by(map_index=map_index)
        query.delete()

    @staticmethod
    def serialize_value(value: Any, *, key: str | None=None, task_id: str | None=None, dag_id: str | None=None, run_id: str | None=None, map_index: int | None=None) -> Any:
        if False:
            while True:
                i = 10
        'Serialize XCom value to str or pickled object.'
        if conf.getboolean('core', 'enable_xcom_pickling'):
            return pickle.dumps(value)
        try:
            return json.dumps(value, cls=XComEncoder).encode('UTF-8')
        except (ValueError, TypeError) as ex:
            log.error('%s. If you are using pickle instead of JSON for XCom, then you need to enable pickle support for XCom in your airflow config or make sure to decorate your object with attr.', ex)
            raise

    @staticmethod
    def _deserialize_value(result: XCom, orm: bool) -> Any:
        if False:
            while True:
                i = 10
        object_hook = None
        if orm:
            object_hook = XComDecoder.orm_object_hook
        if result.value is None:
            return None
        if conf.getboolean('core', 'enable_xcom_pickling'):
            try:
                return pickle.loads(result.value)
            except pickle.UnpicklingError:
                return json.loads(result.value.decode('UTF-8'), cls=XComDecoder, object_hook=object_hook)
        else:
            try:
                return json.loads(result.value.decode('UTF-8'), cls=XComDecoder, object_hook=object_hook)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(result.value)

    @staticmethod
    def deserialize_value(result: XCom) -> Any:
        if False:
            for i in range(10):
                print('nop')
        'Deserialize XCom value from str or pickle object.'
        return BaseXCom._deserialize_value(result, False)

    def orm_deserialize_value(self) -> Any:
        if False:
            return 10
        '\n        Deserialize method which is used to reconstruct ORM XCom object.\n\n        This method should be overridden in custom XCom backends to avoid\n        unnecessary request or other resource consuming operations when\n        creating XCom orm model. This is used when viewing XCom listing\n        in the webserver, for example.\n        '
        return BaseXCom._deserialize_value(self, True)

class _LazyXComAccessIterator(collections.abc.Iterator):

    def __init__(self, cm: contextlib.AbstractContextManager[Query]) -> None:
        if False:
            return 10
        self._cm = cm
        self._entered = False

    def __del__(self) -> None:
        if False:
            return 10
        if self._entered:
            self._cm.__exit__(None, None, None)

    def __iter__(self) -> collections.abc.Iterator:
        if False:
            while True:
                i = 10
        return self

    def __next__(self) -> Any:
        if False:
            print('Hello World!')
        return XCom.deserialize_value(next(self._it))

    @cached_property
    def _it(self) -> collections.abc.Iterator:
        if False:
            for i in range(10):
                print('nop')
        self._entered = True
        return iter(self._cm.__enter__())

@attr.define(slots=True)
class LazyXComAccess(collections.abc.Sequence):
    """Wrapper to lazily pull XCom with a sequence-like interface.

    Note that since the session bound to the parent query may have died when we
    actually access the sequence's content, we must create a new session
    for every function call with ``with_session()``.

    :meta private:
    """
    _query: Query
    _len: int | None = attr.ib(init=False, default=None)

    @classmethod
    def build_from_xcom_query(cls, query: Query) -> LazyXComAccess:
        if False:
            i = 10
            return i + 15
        return cls(query=query.with_entities(XCom.value))

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'LazyXComAccess([{len(self)} items])'

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return str(list(self))

    def __eq__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        if isinstance(other, (list, LazyXComAccess)):
            z = itertools.zip_longest(iter(self), iter(other), fillvalue=object())
            return all((x == y for (x, y) in z))
        return NotImplemented

    def __getstate__(self) -> Any:
        if False:
            return 10
        with self._get_bound_query() as query:
            statement = query.statement.compile(query.session.get_bind(), compile_kwargs={'literal_binds': True})
            return (str(statement), query.count())

    def __setstate__(self, state: Any) -> None:
        if False:
            return 10
        (statement, self._len) = state
        self._query = Query(XCom.value).from_statement(text(statement))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        if self._len is None:
            with self._get_bound_query() as query:
                self._len = query.count()
        return self._len

    def __iter__(self):
        if False:
            while True:
                i = 10
        return _LazyXComAccessIterator(self._get_bound_query())

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        if not isinstance(key, int):
            raise ValueError('only support index access for now')
        try:
            with self._get_bound_query() as query:
                r = query.offset(key).limit(1).one()
        except NoResultFound:
            raise IndexError(key) from None
        return XCom.deserialize_value(r)

    @contextlib.contextmanager
    def _get_bound_query(self) -> Generator[Query, None, None]:
        if False:
            while True:
                i = 10
        if self._query.session and self._query.session.is_active:
            yield self._query
            return
        Session = getattr(settings, 'Session', None)
        if Session is None:
            raise RuntimeError('Session must be set before!')
        session = Session()
        try:
            yield self._query.with_session(session)
        finally:
            session.close()

def _patch_outdated_serializer(clazz: type[BaseXCom], params: Iterable[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Patch a custom ``serialize_value`` to accept the modern signature.\n\n    To give custom XCom backends more flexibility with how they store values, we\n    now forward all params passed to ``XCom.set`` to ``XCom.serialize_value``.\n    In order to maintain compatibility with custom XCom backends written with\n    the old signature, we check the signature and, if necessary, patch with a\n    method that ignores kwargs the backend does not accept.\n    '
    old_serializer = clazz.serialize_value

    @wraps(old_serializer)
    def _shim(**kwargs):
        if False:
            while True:
                i = 10
        kwargs = {k: kwargs.get(k) for k in params}
        warnings.warn(f'Method `serialize_value` in XCom backend {XCom.__name__} is using outdated signature andmust be updated to accept all params in `BaseXCom.set` except `session`. Support will be removed in a future release.', RemovedInAirflow3Warning)
        return old_serializer(**kwargs)
    clazz.serialize_value = _shim

def _get_function_params(function) -> list[str]:
    if False:
        return 10
    '\n    Return the list of variables names of a function.\n\n    :param function: The function to inspect\n    '
    parameters = inspect.signature(function).parameters
    bound_arguments = [name for (name, p) in parameters.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    return bound_arguments

def resolve_xcom_backend() -> type[BaseXCom]:
    if False:
        for i in range(10):
            print('nop')
    'Resolve custom XCom class.\n\n    Confirm that custom XCom class extends the BaseXCom.\n    Compare the function signature of the custom XCom serialize_value to the base XCom serialize_value.\n    '
    clazz = conf.getimport('core', 'xcom_backend', fallback=f'airflow.models.xcom.{BaseXCom.__name__}')
    if not clazz:
        return BaseXCom
    if not issubclass(clazz, BaseXCom):
        raise TypeError(f'Your custom XCom class `{clazz.__name__}` is not a subclass of `{BaseXCom.__name__}`.')
    base_xcom_params = _get_function_params(BaseXCom.serialize_value)
    xcom_params = _get_function_params(clazz.serialize_value)
    if set(base_xcom_params) != set(xcom_params):
        _patch_outdated_serializer(clazz=clazz, params=xcom_params)
    return clazz
if TYPE_CHECKING:
    XCom = BaseXCom
else:
    XCom = resolve_xcom_backend()