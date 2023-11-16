from __future__ import annotations
import os
import statistics
import textwrap
from time import monotonic, sleep
from typing import NamedTuple
import pandas as pd
from airflow.jobs.job import Job, run_job
DAG_FOLDER = os.path.join(os.path.dirname(__file__), 'dags')
os.environ['AIRFLOW__CORE__DAGS_FOLDER'] = DAG_FOLDER
os.environ['AIRFLOW__DEBUG__SQLALCHEMY_STATS'] = 'True'
os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'
LOG_LEVEL = 'INFO'
LOG_FILE = '/files/sql_stats.log'
os.environ['AIRFLOW__LOGGING__LOGGING_CONFIG_CLASS'] = 'scripts.perf.sql_queries.DEBUG_LOGGING_CONFIG'
DEBUG_LOGGING_CONFIG = {'version': 1, 'disable_existing_loggers': False, 'formatters': {'airflow': {'format': '%(message)s'}}, 'handlers': {'console': {'class': 'logging.StreamHandler'}, 'task': {'class': 'logging.FileHandler', 'formatter': 'airflow', 'filename': LOG_FILE}, 'processor': {'class': 'logging.FileHandler', 'formatter': 'airflow', 'filename': LOG_FILE}}, 'loggers': {'airflow.processor': {'handlers': ['processor'], 'level': LOG_LEVEL, 'propagate': False}, 'airflow.task': {'handlers': ['task'], 'level': LOG_LEVEL, 'propagate': False}, 'flask_appbuilder': {'handler': ['console'], 'level': LOG_LEVEL, 'propagate': True}}, 'root': {'handlers': ['console', 'task'], 'level': LOG_LEVEL}}

class Query(NamedTuple):
    """
    Define attributes of the queries that will be picked up by the performance tests.
    """
    function: str
    file: str
    location: int
    sql: str
    stack: str
    time: float

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.function} in {self.file}:{self.location}: {textwrap.shorten(self.sql, 110)}'

    def __eq__(self, other):
        if False:
            return 10
        '\n        Override the __eq__ method to compare specific Query attributes\n        '
        return self.function == other.function and self.sql == other.sql and (self.location == other.location) and (self.file == other.file)

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        '\n        Convert selected attributes of the instance into a dictionary.\n        '
        return dict(zip(('function', 'file', 'location', 'sql', 'stack', 'time'), self))

def reset_db():
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrapper function that calls the airflow resetdb function.\n    '
    from airflow.utils.db import resetdb
    resetdb()

def run_scheduler_job(with_db_reset=False) -> None:
    if False:
        print('Hello World!')
    '\n    Run the scheduler job, selectively resetting the db before creating a ScheduleJob instance\n    '
    from airflow.jobs.scheduler_job_runner import SchedulerJobRunner
    if with_db_reset:
        reset_db()
    job_runner = SchedulerJobRunner(job=Job(), subdir=DAG_FOLDER, do_pickle=False, num_runs=3)
    run_job(job=job_runner.job, execute_callable=job_runner._execute)

def is_query(line: str) -> bool:
    if False:
        while True:
            i = 10
    '\n    Return True, if provided line embeds a query, else False\n    '
    return '@SQLALCHEMY' in line and '|$' in line

def make_report() -> list[Query]:
    if False:
        print('Hello World!')
    '\n    Returns a list of Query objects that are expected to be run during the performance run.\n    '
    queries = []
    with open(LOG_FILE, 'r+') as f:
        raw_queries = [line for line in f.readlines() if is_query(line)]
    for query in raw_queries:
        (time, info, stack, sql) = query.replace('@SQLALCHEMY ', '').split('|$')
        (func, file, loc) = info.split(':')
        file_name = file.rpartition('/')[-1]
        queries.append(Query(function=func.strip(), file=file_name.strip(), location=int(loc.strip()), sql=sql.strip(), stack=stack.strip(), time=float(time.strip())))
    return queries

def run_test() -> tuple[list[Query], float]:
    if False:
        print('Hello World!')
    '\n    Run the tests inside a scheduler and then return the elapsed time along with\n    the queries that will be run.\n    '
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    tic = monotonic()
    run_scheduler_job(with_db_reset=False)
    toc = monotonic()
    queries = make_report()
    return (queries, toc - tic)

def rows_to_csv(rows: list[dict], name: str | None=None) -> pd.DataFrame:
    if False:
        return 10
    '\n    Write results stats to a file.\n    '
    df = pd.DataFrame(rows)
    name = name or f'/files/sql_stats_{int(monotonic())}.csv'
    df.to_csv(name, index=False)
    print(f'Saved result to {name}')
    return df

def main() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Run the tests and write stats to a csv file.\n    '
    reset_db()
    rows = []
    times = []
    for test_no in range(4):
        sleep(5)
        (queries, exec_time) = run_test()
        if test_no:
            times.append(exec_time)
            for qry in queries:
                info = qry.to_dict()
                info['test_no'] = test_no
                rows.append(info)
    rows_to_csv(rows, name='/files/sql_after_remote.csv')
    print(times)
    msg = 'Time for %d dag runs: %.4fs'
    if len(times) > 1:
        print((msg + ' (±%.3fs)') % (len(times), statistics.mean(times), statistics.stdev(times)))
    else:
        print(msg % (len(times), times[0]))
if __name__ == '__main__':
    main()