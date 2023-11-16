from functools import partial
import operator
from peewee import *
from playhouse.db_url import connect as db_url_connect
from huey.api import Huey
from huey.constants import EmptyData
from huey.exceptions import ConfigurationError
from huey.storage import BaseStorage

class BytesBlobField(BlobField):

    def python_value(self, value):
        if False:
            print('Hello World!')
        return value if isinstance(value, bytes) else bytes(value)

class SqlStorage(BaseStorage):

    def __init__(self, name='huey', database=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(SqlStorage, self).__init__(name)
        if database is None:
            raise ConfigurationError('Use of SqlStorage requires a database= argument, which should be a peewee database or a connection string.')
        if isinstance(database, Database):
            self.database = database
        else:
            self.database = db_url_connect(database)
        (self.KV, self.Schedule, self.Task) = self.create_models()
        self.create_tables()

    def create_models(self):
        if False:
            i = 10
            return i + 15

        class Base(Model):

            class Meta:
                database = self.database

        class KV(Base):
            queue = CharField()
            key = CharField()
            value = BytesBlobField()

            class Meta:
                primary_key = CompositeKey('queue', 'key')

        class Schedule(Base):
            queue = CharField()
            data = BytesBlobField()
            timestamp = TimestampField(resolution=1000)

            class Meta:
                indexes = ((('queue', 'timestamp'), False),)

        class Task(Base):
            queue = CharField()
            data = BytesBlobField()
            priority = FloatField(default=0.0)
        Task.add_index(Task.priority.desc(), Task.id)
        return (KV, Schedule, Task)

    def create_tables(self):
        if False:
            i = 10
            return i + 15
        with self.database:
            self.database.create_tables([self.KV, self.Schedule, self.Task])

    def drop_tables(self):
        if False:
            i = 10
            return i + 15
        with self.database:
            self.database.drop_tables([self.KV, self.Schedule, self.Task])

    def close(self):
        if False:
            print('Hello World!')
        return self.database.close()

    def tasks(self, *columns):
        if False:
            while True:
                i = 10
        return self.Task.select(*columns).where(self.Task.queue == self.name)

    def schedule(self, *columns):
        if False:
            for i in range(10):
                print('nop')
        return self.Schedule.select(*columns).where(self.Schedule.queue == self.name)

    def kv(self, *columns):
        if False:
            for i in range(10):
                print('nop')
        return self.KV.select(*columns).where(self.KV.queue == self.name)

    def check_conn(self):
        if False:
            return 10
        if not self.database.is_connection_usable():
            self.database.close()
            self.database.connect()

    def enqueue(self, data, priority=None):
        if False:
            return 10
        self.check_conn()
        self.Task.create(queue=self.name, data=data, priority=priority or 0)

    def dequeue(self):
        if False:
            i = 10
            return i + 15
        self.check_conn()
        query = self.tasks(self.Task.id, self.Task.data).order_by(self.Task.priority.desc(), self.Task.id).limit(1)
        if self.database.for_update:
            query = query.for_update()
        with self.database.atomic():
            try:
                task = query.get()
            except self.Task.DoesNotExist:
                return
            nrows = self.Task.delete().where(self.Task.id == task.id).execute()
            if nrows == 1:
                return task.data

    def queue_size(self):
        if False:
            return 10
        return self.tasks().count()

    def enqueued_items(self, limit=None):
        if False:
            while True:
                i = 10
        query = self.tasks(self.Task.data).order_by(self.Task.priority.desc(), self.Task.id)
        if limit is not None:
            query = query.limit(limit)
        return list(map(operator.itemgetter(0), query.tuples()))

    def flush_queue(self):
        if False:
            while True:
                i = 10
        self.Task.delete().where(self.Task.queue == self.name).execute()

    def add_to_schedule(self, data, timestamp, utc):
        if False:
            print('Hello World!')
        self.check_conn()
        self.Schedule.create(queue=self.name, data=data, timestamp=timestamp)

    def read_schedule(self, timestamp):
        if False:
            i = 10
            return i + 15
        self.check_conn()
        query = self.schedule(self.Schedule.id, self.Schedule.data).where(self.Schedule.timestamp <= timestamp).tuples()
        if self.database.for_update:
            query = query.for_update()
        with self.database.atomic():
            results = list(query)
            if not results:
                return []
            (id_list, data) = zip(*results)
            self.Schedule.delete().where(self.Schedule.id.in_(id_list)).execute()
            return list(data)

    def schedule_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self.schedule().count()

    def scheduled_items(self):
        if False:
            i = 10
            return i + 15
        tasks = self.schedule(self.Schedule.data).order_by(self.Schedule.timestamp).tuples()
        return list(map(operator.itemgetter(0), tasks))

    def flush_schedule(self):
        if False:
            while True:
                i = 10
        self.Schedule.delete().where(self.Schedule.queue == self.name).execute()

    def put_data(self, key, value, is_result=False):
        if False:
            i = 10
            return i + 15
        self.check_conn()
        if isinstance(self.database, PostgresqlDatabase):
            self.KV.insert(queue=self.name, key=key, value=value).on_conflict(conflict_target=[self.KV.queue, self.KV.key], preserve=[self.KV.value]).execute()
        else:
            self.KV.replace(queue=self.name, key=key, value=value).execute()

    def peek_data(self, key):
        if False:
            i = 10
            return i + 15
        self.check_conn()
        try:
            kv = self.kv(self.KV.value).where(self.KV.key == key).get()
        except self.KV.DoesNotExist:
            return EmptyData
        else:
            return kv.value

    def pop_data(self, key):
        if False:
            while True:
                i = 10
        self.check_conn()
        query = self.kv().where(self.KV.key == key)
        if self.database.for_update:
            query = query.for_update()
        with self.database.atomic():
            try:
                kv = query.get()
            except self.KV.DoesNotExist:
                return EmptyData
            else:
                dq = self.KV.delete().where((self.KV.queue == self.name) & (self.KV.key == key))
                return kv.value if dq.execute() == 1 else EmptyData

    def has_data_for_key(self, key):
        if False:
            while True:
                i = 10
        self.check_conn()
        return self.kv().where(self.KV.key == key).exists()

    def put_if_empty(self, key, value):
        if False:
            print('Hello World!')
        self.check_conn()
        try:
            with self.database.atomic():
                self.KV.insert(queue=self.name, key=key, value=value).execute()
        except IntegrityError:
            return False
        else:
            return True

    def result_store_size(self):
        if False:
            while True:
                i = 10
        return self.kv().count()

    def result_items(self):
        if False:
            print('Hello World!')
        query = self.kv(self.KV.key, self.KV.value).tuples()
        return dict(((k, v) for (k, v) in query.iterator()))

    def flush_results(self):
        if False:
            for i in range(10):
                print('nop')
        self.KV.delete().where(self.KV.queue == self.name).execute()
SqlHuey = partial(Huey, storage_class=SqlStorage)