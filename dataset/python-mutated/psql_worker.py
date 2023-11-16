import csv
import datetime
import os
from subprocess import PIPE
from wal_e.piper import popen_nonblock
from wal_e.exception import UserException
PSQL_BIN = 'psql'

class UTC(datetime.tzinfo):
    """
    UTC timezone

    Adapted from a Python example

    """
    ZERO = datetime.timedelta(0)
    HOUR = datetime.timedelta(hours=1)

    def utcoffset(self, dt):
        if False:
            i = 10
            return i + 15
        return self.ZERO

    def tzname(self, dt):
        if False:
            while True:
                i = 10
        return 'UTC'

    def dst(self, dt):
        if False:
            while True:
                i = 10
        return self.ZERO

def psql_csv_run(sql_command, error_handler=None):
    if False:
        while True:
            i = 10
    '\n    Runs psql and returns a CSVReader object from the query\n\n    This CSVReader includes header names as the first record in all\n    situations.  The output is fully buffered into Python.\n\n    '
    csv_query = 'COPY ({query}) TO STDOUT WITH CSV HEADER;'.format(query=sql_command)
    new_env = os.environ.copy()
    new_env.setdefault('PGOPTIONS', '')
    new_env['PGOPTIONS'] += ' --statement-timeout=0'
    psql_proc = popen_nonblock([PSQL_BIN, '-d', 'postgres', '--no-password', '--no-psqlrc', '-c', csv_query], stdout=PIPE, env=new_env)
    stdout = psql_proc.communicate()[0].decode('utf-8')
    if psql_proc.returncode != 0:
        if error_handler is not None:
            error_handler(psql_proc)
        else:
            assert error_handler is None
            raise UserException('could not csv-execute a query successfully via psql', 'Query was "{query}".'.format(sql_command), 'You may have to set some libpq environment variables if you are sure the server is running.')
    assert psql_proc.returncode == 0
    return csv.reader(iter(stdout.strip().split('\n')))

class PgBackupStatements(object):
    """
    Contains operators to start and stop a backup on a Postgres server

    Relies on PsqlHelp for underlying mechanism.

    """
    _WAL_NAME = None

    @staticmethod
    def _dict_transform(csv_reader):
        if False:
            return 10
        rows = list(csv_reader)
        assert len(rows) == 2, 'Expect header row and data row'
        return dict(list(zip(*rows)))

    @classmethod
    def _wal_name(cls):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets and returns _WAL_NAME to 'wal' or 'xlog' depending on\n        version of postgres we are working with.\n\n        It is used for handling xlog -> wal rename in postgres v10\n\n        "
        if cls._WAL_NAME is None:
            version = cls._dict_transform(psql_csv_run("SELECT current_setting('server_version_num')"))
            if int(version['current_setting']) >= 100000:
                cls._WAL_NAME = 'wal'
            else:
                cls._WAL_NAME = 'xlog'
        return cls._WAL_NAME

    @classmethod
    def run_start_backup(cls):
        if False:
            while True:
                i = 10
        '\n        Connects to a server and attempts to start a hot backup\n\n        Yields the WAL information in a dictionary for bookkeeping and\n        recording.\n\n        '

        def handler(popen):
            if False:
                for i in range(10):
                    print('nop')
            assert popen.returncode != 0
            raise UserException('Could not start hot backup')
        label = 'freeze_start_' + datetime.datetime.utcnow().replace(tzinfo=UTC()).isoformat()
        return cls._dict_transform(psql_csv_run("SELECT file_name,   lpad(file_offset::text, 8, '0') AS file_offset FROM pg_{0}file_name_offset(  pg_start_backup('{1}'))".format(cls._wal_name(), label), error_handler=handler))

    @classmethod
    def run_stop_backup(cls):
        if False:
            print('Hello World!')
        '\n        Stop a hot backup, if it was running, or error\n\n        Return the last WAL file name and position that is required to\n        gain consistency on the captured heap.\n\n        '

        def handler(popen):
            if False:
                i = 10
                return i + 15
            assert popen.returncode != 0
            raise UserException('Could not stop hot backup')
        return cls._dict_transform(psql_csv_run("SELECT file_name,   lpad(file_offset::text, 8, '0') AS file_offset FROM pg_{0}file_name_offset(  pg_stop_backup())".format(cls._wal_name()), error_handler=handler))

    @classmethod
    def pg_version(cls):
        if False:
            print('Hello World!')
        '\n        Get a very informative version string from Postgres\n\n        Includes minor version, major version, and architecture, among\n        other details.\n\n        '
        return cls._dict_transform(psql_csv_run('SELECT * FROM version()'))