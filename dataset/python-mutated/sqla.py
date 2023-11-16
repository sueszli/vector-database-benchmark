"""
Support for SQLAlchemy. Provides SQLAlchemyTarget for storing in databases
supported by SQLAlchemy. The user would be responsible for installing the
required database driver to connect using SQLAlchemy.

Minimal example of a job to copy data to database using SQLAlchemy is as shown
below:

.. code-block:: python

    from sqlalchemy import String
    import luigi
    from luigi.contrib import sqla

    class SQLATask(sqla.CopyToTable):
        # columns defines the table schema, with each element corresponding
        # to a column in the format (args, kwargs) which will be sent to
        # the sqlalchemy.Column(*args, **kwargs)
        columns = [
            (["item", String(64)], {"primary_key": True}),
            (["property", String(64)], {})
        ]
        connection_string = "sqlite://"  # in memory SQLite database
        table = "item_property"  # name of the table to store data

        def rows(self):
            for row in [("item1", "property1"), ("item2", "property2")]:
                yield row

    if __name__ == '__main__':
        task = SQLATask()
        luigi.build([task], local_scheduler=True)


If the target table where the data needs to be copied already exists, then
the column schema definition can be skipped and instead the reflect flag
can be set as True. Here is a modified version of the above example:

.. code-block:: python

    from sqlalchemy import String
    import luigi
    from luigi.contrib import sqla

    class SQLATask(sqla.CopyToTable):
        # If database table is already created, then the schema can be loaded
        # by setting the reflect flag to True
        reflect = True
        connection_string = "sqlite://"  # in memory SQLite database
        table = "item_property"  # name of the table to store data

        def rows(self):
            for row in [("item1", "property1"), ("item2", "property2")]:
                yield row

    if __name__ == '__main__':
        task = SQLATask()
        luigi.build([task], local_scheduler=True)


In the above examples, the data that needs to be copied was directly provided by
overriding the rows method. Alternately, if the data comes from another task, the
modified example would look as shown below:

.. code-block:: python

    from sqlalchemy import String
    import luigi
    from luigi.contrib import sqla
    from luigi.mock import MockTarget

    class BaseTask(luigi.Task):
        def output(self):
            return MockTarget("BaseTask")

        def run(self):
            out = self.output().open("w")
            TASK_LIST = ["item%d\\tproperty%d\\n" % (i, i) for i in range(10)]
            for task in TASK_LIST:
                out.write(task)
            out.close()

    class SQLATask(sqla.CopyToTable):
        # columns defines the table schema, with each element corresponding
        # to a column in the format (args, kwargs) which will be sent to
        # the sqlalchemy.Column(*args, **kwargs)
        columns = [
            (["item", String(64)], {"primary_key": True}),
            (["property", String(64)], {})
        ]
        connection_string = "sqlite://"  # in memory SQLite database
        table = "item_property"  # name of the table to store data

        def requires(self):
            return BaseTask()

    if __name__ == '__main__':
        task1, task2 = SQLATask(), BaseTask()
        luigi.build([task1, task2], local_scheduler=True)


In the above example, the output from `BaseTask` is copied into the
database. Here we did not have to implement the `rows` method because
by default `rows` implementation assumes every line is a row with
column values separated by a tab. One can define `column_separator`
option for the task if the values are say comma separated instead of
tab separated.

You can pass in database specific connection arguments by setting the connect_args
dictionary.  The options will be passed directly to the DBAPI's connect method as
keyword arguments.

The other option to `sqla.CopyToTable` that can be of help with performance aspect is the
`chunk_size`. The default is 5000. This is the number of rows that will be inserted in
a transaction at a time. Depending on the size of the inserts, this value can be tuned
for performance.

See here for a `tutorial on building task pipelines using luigi
<http://gouthamanbalaraman.com/blog/building-luigi-task-pipeline.html>`_ and
using `SQLAlchemy in workflow pipelines <http://gouthamanbalaraman.com/blog/sqlalchemy-luigi-workflow-pipeline.html>`_.

Author: Gouthaman Balaraman
Date: 01/02/2015
"""
import abc
import collections
import datetime
import itertools
import logging
import luigi
import os
import sqlalchemy

class SQLAlchemyTarget(luigi.Target):
    """
    Database target using SQLAlchemy.

    This will rarely have to be directly instantiated by the user.

    Typical usage would be to override `luigi.contrib.sqla.CopyToTable` class
    to create a task to write to the database.
    """
    marker_table = None
    _engine_dict = {}
    Connection = collections.namedtuple('Connection', 'engine pid')

    def __init__(self, connection_string, target_table, update_id, echo=False, connect_args=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor for the SQLAlchemyTarget.\n\n        :param connection_string: SQLAlchemy connection string\n        :type connection_string: str\n        :param target_table: The table name for the data\n        :type target_table: str\n        :param update_id: An identifier for this data set\n        :type update_id: str\n        :param echo: Flag to setup SQLAlchemy logging\n        :type echo: bool\n        :param connect_args: A dictionary of connection arguments\n        :type connect_args: dict\n        :return:\n        '
        if connect_args is None:
            connect_args = {}
        self.target_table = target_table
        self.update_id = update_id
        self.connection_string = connection_string
        self.echo = echo
        self.connect_args = connect_args
        self.marker_table_bound = None

    def __str__(self):
        if False:
            print('Hello World!')
        return self.target_table

    @property
    def engine(self):
        if False:
            i = 10
            return i + 15
        "\n        Return an engine instance, creating it if it doesn't exist.\n\n        Recreate the engine connection if it wasn't originally created\n        by the current process.\n        "
        pid = os.getpid()
        conn = SQLAlchemyTarget._engine_dict.get(self.connection_string)
        if not conn or conn.pid != pid:
            engine = sqlalchemy.create_engine(self.connection_string, connect_args=self.connect_args, echo=self.echo)
            SQLAlchemyTarget._engine_dict[self.connection_string] = self.Connection(engine, pid)
        return SQLAlchemyTarget._engine_dict[self.connection_string].engine

    def touch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mark this update as complete.\n        '
        if self.marker_table_bound is None:
            self.create_marker_table()
        table = self.marker_table_bound
        id_exists = self.exists()
        with self.engine.begin() as conn:
            if not id_exists:
                ins = table.insert().values(update_id=self.update_id, target_table=self.target_table, inserted=datetime.datetime.now())
            else:
                ins = table.update().where(sqlalchemy.and_(table.c.update_id == self.update_id, table.c.target_table == self.target_table)).values(update_id=self.update_id, target_table=self.target_table, inserted=datetime.datetime.now())
            conn.execute(ins)
        assert self.exists()

    def exists(self):
        if False:
            for i in range(10):
                print('nop')
        row = None
        if self.marker_table_bound is None:
            self.create_marker_table()
        with self.engine.begin() as conn:
            table = self.marker_table_bound
            s = sqlalchemy.select([table]).where(sqlalchemy.and_(table.c.update_id == self.update_id, table.c.target_table == self.target_table)).limit(1)
            row = conn.execute(s).fetchone()
        return row is not None

    def create_marker_table(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create marker table if it doesn't exist.\n\n        Using a separate connection since the transaction might have to be reset.\n        "
        if self.marker_table is None:
            self.marker_table = luigi.configuration.get_config().get('sqlalchemy', 'marker-table', 'table_updates')
        engine = self.engine
        with engine.begin() as con:
            metadata = sqlalchemy.MetaData()
            if not con.dialect.has_table(con, self.marker_table):
                self.marker_table_bound = sqlalchemy.Table(self.marker_table, metadata, sqlalchemy.Column('update_id', sqlalchemy.String(128), primary_key=True), sqlalchemy.Column('target_table', sqlalchemy.String(128)), sqlalchemy.Column('inserted', sqlalchemy.DateTime, default=datetime.datetime.now()))
                metadata.create_all(engine)
            else:
                metadata.reflect(only=[self.marker_table], bind=engine)
                self.marker_table_bound = metadata.tables[self.marker_table]

    def open(self, mode):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Cannot open() SQLAlchemyTarget')

class CopyToTable(luigi.Task):
    """
    An abstract task for inserting a data set into SQLAlchemy RDBMS

    Usage:

    * subclass and override the required `connection_string`, `table` and `columns` attributes.
    * optionally override the `schema` attribute to use a different schema for
      the target table.
    """
    _logger = logging.getLogger('luigi-interface')
    echo = False
    connect_args = {}

    @property
    @abc.abstractmethod
    def connection_string(self):
        if False:
            print('Hello World!')
        return None

    @property
    @abc.abstractmethod
    def table(self):
        if False:
            print('Hello World!')
        return None
    columns = []
    schema = ''
    column_separator = '\t'
    chunk_size = 5000
    reflect = False

    def create_table(self, engine):
        if False:
            i = 10
            return i + 15
        '\n        Override to provide code for creating the target table.\n\n        By default it will be created using types specified in columns.\n        If the table exists, then it binds to the existing table.\n\n        If overridden, use the provided connection object for setting up the table in order to\n        create the table and insert data using the same transaction.\n        :param engine: The sqlalchemy engine instance\n        :type engine: object\n        '

        def construct_sqla_columns(columns):
            if False:
                i = 10
                return i + 15
            retval = [sqlalchemy.Column(*c[0], **c[1]) for c in columns]
            return retval
        needs_setup = len(self.columns) == 0 or False in [len(c) == 2 for c in self.columns] if not self.reflect else False
        if needs_setup:
            raise NotImplementedError('create_table() not implemented for %r and columns types not specified' % self.table)
        else:
            with engine.begin() as con:
                if self.schema:
                    metadata = sqlalchemy.MetaData(schema=self.schema)
                else:
                    metadata = sqlalchemy.MetaData()
                try:
                    if not con.dialect.has_table(con, self.table, self.schema or None):
                        sqla_columns = construct_sqla_columns(self.columns)
                        self.table_bound = sqlalchemy.Table(self.table, metadata, *sqla_columns)
                        metadata.create_all(engine)
                    else:
                        full_table = '.'.join([self.schema, self.table]) if self.schema else self.table
                        metadata.reflect(only=[self.table], bind=engine)
                        self.table_bound = metadata.tables[full_table]
                except Exception as e:
                    self._logger.exception(self.table + str(e))

    def update_id(self):
        if False:
            print('Hello World!')
        '\n        This update id will be a unique identifier for this insert on this table.\n        '
        return self.task_id

    def output(self):
        if False:
            for i in range(10):
                print('nop')
        return SQLAlchemyTarget(connection_string=self.connection_string, target_table=self.table, update_id=self.update_id(), connect_args=self.connect_args, echo=self.echo)

    def rows(self):
        if False:
            print('Hello World!')
        '\n        Return/yield tuples or lists corresponding to each row to be inserted.\n\n        This method can be overridden for custom file types or formats.\n        '
        with self.input().open('r') as fobj:
            for line in fobj:
                yield line.strip('\n').split(self.column_separator)

    def run(self):
        if False:
            while True:
                i = 10
        self._logger.info('Running task copy to table for update id %s for table %s' % (self.update_id(), self.table))
        output = self.output()
        engine = output.engine
        self.create_table(engine)
        with engine.begin() as conn:
            rows = iter(self.rows())
            ins_rows = [dict(zip(('_' + c.key for c in self.table_bound.c), row)) for row in itertools.islice(rows, self.chunk_size)]
            while ins_rows:
                self.copy(conn, ins_rows, self.table_bound)
                ins_rows = [dict(zip(('_' + c.key for c in self.table_bound.c), row)) for row in itertools.islice(rows, self.chunk_size)]
                self._logger.info('Finished inserting %d rows into SQLAlchemy target' % len(ins_rows))
        output.touch()
        self._logger.info('Finished inserting rows into SQLAlchemy target')

    def copy(self, conn, ins_rows, table_bound):
        if False:
            while True:
                i = 10
        '\n        This method does the actual insertion of the rows of data given by ins_rows into the\n        database. A task that needs row updates instead of insertions should overload this method.\n        :param conn: The sqlalchemy connection object\n        :param ins_rows: The dictionary of rows with the keys in the format _<column_name>. For example\n        if you have a table with a column name "property", then the key in the dictionary\n        would be "_property". This format is consistent with the bindparam usage in sqlalchemy.\n        :param table_bound: The object referring to the table\n        :return:\n        '
        bound_cols = dict(((c, sqlalchemy.bindparam('_' + c.key)) for c in table_bound.columns))
        ins = table_bound.insert().values(bound_cols)
        conn.execute(ins, ins_rows)