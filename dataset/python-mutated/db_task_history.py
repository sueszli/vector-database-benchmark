"""
Provides a database backend to the central scheduler. This lets you see historical runs.
See :ref:`TaskHistory` for information about how to turn out the task history feature.
"""
import datetime
import logging
from contextlib import contextmanager
from luigi import configuration
from luigi import task_history
from luigi.task_status import DONE, FAILED, PENDING, RUNNING
import sqlalchemy
import sqlalchemy.ext.declarative
import sqlalchemy.orm
import sqlalchemy.orm.collections
from sqlalchemy.engine import reflection
Base = sqlalchemy.ext.declarative.declarative_base()
logger = logging.getLogger('luigi-interface')

class DbTaskHistory(task_history.TaskHistory):
    """
    Task History that writes to a database using sqlalchemy.
    Also has methods for useful db queries.
    """
    CURRENT_SOURCE_VERSION = 1

    @contextmanager
    def _session(self, session=None):
        if False:
            return 10
        if session:
            yield session
        else:
            session = self.session_factory()
            try:
                yield session
            except BaseException:
                session.rollback()
                raise
            else:
                session.commit()

    def __init__(self):
        if False:
            while True:
                i = 10
        config = configuration.get_config()
        connection_string = config.get('task_history', 'db_connection')
        self.engine = sqlalchemy.create_engine(connection_string)
        self.session_factory = sqlalchemy.orm.sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        self.tasks = {}
        _upgrade_schema(self.engine)

    def task_scheduled(self, task):
        if False:
            while True:
                i = 10
        htask = self._get_task(task, status=PENDING)
        self._add_task_event(htask, TaskEvent(event_name=PENDING, ts=datetime.datetime.now()))

    def task_finished(self, task, successful):
        if False:
            print('Hello World!')
        event_name = DONE if successful else FAILED
        htask = self._get_task(task, status=event_name)
        self._add_task_event(htask, TaskEvent(event_name=event_name, ts=datetime.datetime.now()))

    def task_started(self, task, worker_host):
        if False:
            for i in range(10):
                print('nop')
        htask = self._get_task(task, status=RUNNING, host=worker_host)
        self._add_task_event(htask, TaskEvent(event_name=RUNNING, ts=datetime.datetime.now()))

    def _get_task(self, task, status, host=None):
        if False:
            for i in range(10):
                print('nop')
        if task.id in self.tasks:
            htask = self.tasks[task.id]
            htask.status = status
            if host:
                htask.host = host
        else:
            htask = self.tasks[task.id] = task_history.StoredTask(task, status, host)
        return htask

    def _add_task_event(self, task, event):
        if False:
            i = 10
            return i + 15
        for (task_record, session) in self._find_or_create_task(task):
            task_record.events.append(event)

    def _find_or_create_task(self, task):
        if False:
            return 10
        with self._session() as session:
            if task.record_id is not None:
                logger.debug('Finding task with record_id [%d]', task.record_id)
                task_record = session.query(TaskRecord).get(task.record_id)
                if not task_record:
                    raise Exception('Task with record_id, but no matching Task record!')
                yield (task_record, session)
            else:
                task_record = TaskRecord(task_id=task._task.id, name=task.task_family, host=task.host)
                for (k, v) in task.parameters.items():
                    task_record.parameters[k] = TaskParameter(name=k, value=v)
                session.add(task_record)
                yield (task_record, session)
            if task.host:
                task_record.host = task.host
        task.record_id = task_record.id

    def find_all_by_parameters(self, task_name, session=None, **task_params):
        if False:
            i = 10
            return i + 15
        '\n        Find tasks with the given task_name and the same parameters as the kwargs.\n        '
        with self._session(session) as session:
            query = session.query(TaskRecord).join(TaskEvent).filter(TaskRecord.name == task_name)
            for (k, v) in task_params.items():
                alias = sqlalchemy.orm.aliased(TaskParameter)
                query = query.join(alias).filter(alias.name == k, alias.value == v)
            tasks = query.order_by(TaskEvent.ts)
            for task in tasks:
                assert all((k in task.parameters and v == str(task.parameters[k].value) for (k, v) in task_params.items()))
                yield task

    def find_all_by_name(self, task_name, session=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find all tasks with the given task_name.\n        '
        return self.find_all_by_parameters(task_name, session)

    def find_latest_runs(self, session=None):
        if False:
            i = 10
            return i + 15
        '\n        Return tasks that have been updated in the past 24 hours.\n        '
        with self._session(session) as session:
            yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
            return session.query(TaskRecord).join(TaskEvent).filter(TaskEvent.ts >= yesterday).group_by(TaskRecord.id, TaskEvent.event_name, TaskEvent.ts).order_by(TaskEvent.ts.desc()).all()

    def find_all_runs(self, session=None):
        if False:
            return 10
        '\n        Return all tasks that have been updated.\n        '
        with self._session(session) as session:
            return session.query(TaskRecord).all()

    def find_all_events(self, session=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return all running/failed/done events.\n        '
        with self._session(session) as session:
            return session.query(TaskEvent).all()

    def find_task_by_id(self, id, session=None):
        if False:
            return 10
        '\n        Find task with the given record ID.\n        '
        with self._session(session) as session:
            return session.query(TaskRecord).get(id)

class TaskParameter(Base):
    """
    Table to track luigi.Parameter()s of a Task.
    """
    __tablename__ = 'task_parameters'
    task_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey('tasks.id'), primary_key=True)
    name = sqlalchemy.Column(sqlalchemy.String(128), primary_key=True)
    value = sqlalchemy.Column(sqlalchemy.Text())

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'TaskParameter(task_id=%d, name=%s, value=%s)' % (self.task_id, self.name, self.value)

class TaskEvent(Base):
    """
    Table to track when a task is scheduled, starts, finishes, and fails.
    """
    __tablename__ = 'task_events'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    task_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey('tasks.id'), index=True)
    event_name = sqlalchemy.Column(sqlalchemy.String(20))
    ts = sqlalchemy.Column(sqlalchemy.TIMESTAMP, index=True, nullable=False)

    def __repr__(self):
        if False:
            return 10
        return 'TaskEvent(task_id=%s, event_name=%s, ts=%s' % (self.task_id, self.event_name, self.ts)

class TaskRecord(Base):
    """
    Base table to track information about a luigi.Task.

    References to other tables are available through task.events, task.parameters, etc.
    """
    __tablename__ = 'tasks'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    task_id = sqlalchemy.Column(sqlalchemy.String(200), index=True)
    name = sqlalchemy.Column(sqlalchemy.String(128), index=True)
    host = sqlalchemy.Column(sqlalchemy.String(128))
    parameters = sqlalchemy.orm.relationship('TaskParameter', collection_class=sqlalchemy.orm.collections.attribute_mapped_collection('name'), cascade='all, delete-orphan')
    events = sqlalchemy.orm.relationship('TaskEvent', order_by=(sqlalchemy.desc(TaskEvent.ts), sqlalchemy.desc(TaskEvent.id)), backref='task')

    def __repr__(self):
        if False:
            return 10
        return 'TaskRecord(name=%s, host=%s)' % (self.name, self.host)

def _upgrade_schema(engine):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure the database schema is up to date with the codebase.\n\n    :param engine: SQLAlchemy engine of the underlying database.\n    '
    inspector = reflection.Inspector.from_engine(engine)
    with engine.connect() as conn:
        if 'task_id' not in [x['name'] for x in inspector.get_columns('tasks')]:
            logger.warning('Upgrading DbTaskHistory schema: Adding tasks.task_id')
            conn.execute('ALTER TABLE tasks ADD COLUMN task_id VARCHAR(200)')
            conn.execute('CREATE INDEX ix_task_id ON tasks (task_id)')
        if 'mysql' in engine.dialect.name:
            conn.execute('ALTER TABLE task_parameters MODIFY COLUMN value TEXT')
        elif 'oracle' in engine.dialect.name:
            conn.execute('ALTER TABLE task_parameters MODIFY value TEXT')
        elif 'mssql' in engine.dialect.name:
            conn.execute('ALTER TABLE task_parameters ALTER COLUMN value TEXT')
        elif 'postgresql' in engine.dialect.name:
            if str([x for x in inspector.get_columns('task_parameters') if x['name'] == 'value'][0]['type']) != 'TEXT':
                conn.execute('ALTER TABLE task_parameters ALTER COLUMN value TYPE TEXT')
        elif 'sqlite' in engine.dialect.name:
            for i in conn.execute('PRAGMA table_info(task_parameters);').fetchall():
                if i['name'] == 'value' and i['type'] != 'TEXT':
                    logger.warning('SQLite can not change column types. Please use a new database to pickup column type changes.')
        else:
            logger.warning('SQLAlcheny dialect {} could not be migrated to the TEXT type'.format(engine.dialect))