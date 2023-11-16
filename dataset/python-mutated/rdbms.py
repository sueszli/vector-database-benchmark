"""
A common module for postgres like databases, such as postgres or redshift
"""
import abc
import logging
import luigi
import luigi.task
logger = logging.getLogger('luigi-interface')

class _MetadataColumnsMixin:
    """Provide an additional behavior that adds columns and values to tables

    This mixin is used to provide an additional behavior that allow a task to
    add generic metadata columns to every table created for both PSQL and
    Redshift.

    Example:

        This is a use-case example of how this mixin could come handy and how
        to use it.

        .. code:: python

            class CommonMetaColumnsBehavior:
                def update_report_execution_date_query(self):
                    query = "UPDATE {0} "                             "SET date_param = DATE '{1}' "                             "WHERE date_param IS NULL".format(self.table, self.date)

                    return query

                @property
                def metadata_columns(self):
                    if self.date:
                        cols.append(('date_param', 'VARCHAR'))

                    return cols

                @property
                def metadata_queries(self):
                    queries = [self.update_created_tz_query()]
                    if self.date:
                        queries.append(self.update_report_execution_date_query())

                    return queries


            class RedshiftCopyCSVToTableFromS3(CommonMetaColumnsBehavior, redshift.S3CopyToTable):
                "We have some business override here that would only add noise to the
                example, so let's assume that this is only a shell."
                pass


            class UpdateTableA(RedshiftCopyCSVToTableFromS3):
                date = luigi.Parameter()
                table = 'tableA'

                def queries():
                    return [query_content_for('/queries/deduplicate_dupes.sql')]


            class UpdateTableB(RedshiftCopyCSVToTableFromS3):
                date = luigi.Parameter()
                table = 'tableB'
    """

    @property
    def metadata_columns(self):
        if False:
            i = 10
            return i + 15
        'Returns the default metadata columns.\n\n        Those columns are columns that we want each tables to have by default.\n        '
        return []

    @property
    def metadata_queries(self):
        if False:
            while True:
                i = 10
        return []

    @property
    def enable_metadata_columns(self):
        if False:
            while True:
                i = 10
        return False

    def _add_metadata_columns(self, connection):
        if False:
            print('Hello World!')
        cursor = connection.cursor()
        for column in self.metadata_columns:
            if len(column) == 0:
                raise ValueError('_add_metadata_columns is unable to infer column information from column {column} for {table}'.format(column=column, table=self.table))
            column_name = column[0]
            if not self._column_exists(cursor, column_name):
                logger.info('Adding missing metadata column {column} to {table}'.format(column=column, table=self.table))
                self._add_column_to_table(cursor, column)

    def _column_exists(self, cursor, column_name):
        if False:
            while True:
                i = 10
        if '.' in self.table:
            (schema, table) = self.table.split('.')
            query = "SELECT 1 AS column_exists FROM information_schema.columns WHERE table_schema = LOWER('{0}') AND table_name = LOWER('{1}') AND column_name = LOWER('{2}') LIMIT 1;".format(schema, table, column_name)
        else:
            query = "SELECT 1 AS column_exists FROM information_schema.columns WHERE table_name = LOWER('{0}') AND column_name = LOWER('{1}') LIMIT 1;".format(self.table, column_name)
        cursor.execute(query)
        result = cursor.fetchone()
        return bool(result)

    def _add_column_to_table(self, cursor, column):
        if False:
            for i in range(10):
                print('nop')
        if len(column) == 1:
            raise ValueError('_add_column_to_table() column type not specified for {column}'.format(column=column[0]))
        elif len(column) == 2:
            query = 'ALTER TABLE {table} ADD COLUMN {column};'.format(table=self.table, column=' '.join(column))
        elif len(column) == 3:
            query = 'ALTER TABLE {table} ADD COLUMN {column} ENCODE {encoding};'.format(table=self.table, column=' '.join(column[0:2]), encoding=column[2])
        else:
            raise ValueError('_add_column_to_table() found no matching behavior for {column}'.format(column=column))
        cursor.execute(query)

    def post_copy_metacolumns(self, cursor):
        if False:
            print('Hello World!')
        logger.info('Executing post copy metadata queries')
        for query in self.metadata_queries:
            cursor.execute(query)

class CopyToTable(luigi.task.MixinNaiveBulkComplete, _MetadataColumnsMixin, luigi.Task):
    """
    An abstract task for inserting a data set into RDBMS.

    Usage:

        Subclass and override the following attributes:

        * `host`,
        * `database`,
        * `user`,
        * `password`,
        * `table`
        * `columns`
        * `port`
    """

    @property
    @abc.abstractmethod
    def host(self):
        if False:
            print('Hello World!')
        return None

    @property
    @abc.abstractmethod
    def database(self):
        if False:
            return 10
        return None

    @property
    @abc.abstractmethod
    def user(self):
        if False:
            print('Hello World!')
        return None

    @property
    @abc.abstractmethod
    def password(self):
        if False:
            i = 10
            return i + 15
        return None

    @property
    @abc.abstractmethod
    def table(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    @property
    def port(self):
        if False:
            return 10
        return None
    columns = []
    null_values = (None,)
    column_separator = '\t'

    def create_table(self, connection):
        if False:
            while True:
                i = 10
        '\n        Override to provide code for creating the target table.\n\n        By default it will be created using types (optionally) specified in columns.\n\n        If overridden, use the provided connection object for setting up the table in order to\n        create the table and insert data using the same transaction.\n        '
        if len(self.columns[0]) == 1:
            raise NotImplementedError('create_table() not implemented for %r and columns types not specified' % self.table)
        elif len(self.columns[0]) == 2:
            coldefs = ','.join(('{name} {type}'.format(name=name, type=type) for (name, type) in self.columns))
            query = 'CREATE TABLE {table} ({coldefs})'.format(table=self.table, coldefs=coldefs)
            connection.cursor().execute(query)

    @property
    def update_id(self):
        if False:
            print('Hello World!')
        '\n        This update id will be a unique identifier for this insert on this table.\n        '
        return self.task_id

    @abc.abstractmethod
    def output(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('This method must be overridden')

    def init_copy(self, connection):
        if False:
            return 10
        '\n        Override to perform custom queries.\n\n        Any code here will be formed in the same transaction as the main copy, just prior to copying data.\n        Example use cases include truncating the table or removing all data older than X in the database\n        to keep a rolling window of data available in the table.\n        '
        if hasattr(self, 'clear_table'):
            raise Exception('The clear_table attribute has been removed. Override init_copy instead!')
        if self.enable_metadata_columns:
            self._add_metadata_columns(connection)

    def post_copy(self, connection):
        if False:
            i = 10
            return i + 15
        '\n        Override to perform custom queries.\n\n        Any code here will be formed in the same transaction as the main copy, just after copying data.\n        Example use cases include cleansing data in temp table prior to insertion into real table.\n        '
        pass

    @abc.abstractmethod
    def copy(self, cursor, file):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('This method must be overridden')

class Query(luigi.task.MixinNaiveBulkComplete, luigi.Task):
    """
    An abstract task for executing an RDBMS query.

    Usage:

        Subclass and override the following attributes:

        * `host`,
        * `database`,
        * `user`,
        * `password`,
        * `table`,
        * `query`

        Optionally override:

        * `port`,
        * `autocommit`
        * `update_id`

        Subclass and override the following methods:

        * `run`
        * `output`
    """

    @property
    @abc.abstractmethod
    def host(self):
        if False:
            while True:
                i = 10
        '\n        Host of the RDBMS. Implementation should support `hostname:port`\n        to encode port.\n        '
        return None

    @property
    def port(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override to specify port separately from host.\n        '
        return None

    @property
    @abc.abstractmethod
    def database(self):
        if False:
            while True:
                i = 10
        return None

    @property
    @abc.abstractmethod
    def user(self):
        if False:
            while True:
                i = 10
        return None

    @property
    @abc.abstractmethod
    def password(self):
        if False:
            return 10
        return None

    @property
    @abc.abstractmethod
    def table(self):
        if False:
            print('Hello World!')
        return None

    @property
    @abc.abstractmethod
    def query(self):
        if False:
            return 10
        return None

    @property
    def autocommit(self):
        if False:
            print('Hello World!')
        return False

    @property
    def update_id(self):
        if False:
            print('Hello World!')
        "\n        Override to create a custom marker table 'update_id' signature for Query subclass task instances\n        "
        return self.task_id

    @abc.abstractmethod
    def run(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('This method must be overridden')

    @abc.abstractmethod
    def output(self):
        if False:
            while True:
                i = 10
        '\n        Override with an RDBMS Target (e.g. PostgresTarget or RedshiftTarget) to record execution in a marker table\n        '
        raise NotImplementedError('This method must be overridden')