import argparse
import json
import logging
import os
import random
import sqlite3
import subprocess
import time
from enum import Enum
from logging.handlers import RotatingFileHandler
from sqlite3 import Connection
from subprocess import PIPE
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
dot_allennlp_dir = f"{os.environ['HOME']}/.allennlp"
if not os.path.exists(dot_allennlp_dir):
    os.mkdir(dot_allennlp_dir)
handler = RotatingFileHandler(f'{dot_allennlp_dir}/resume.log', maxBytes=1024 * 1024, backupCount=10)
handler.setFormatter(formatter)
logger.addHandler(handler)
BEAKER_QUERY_INTERVAL_SECONDS = 1.0

class BeakerStatus(Enum):
    submitted = 'submitted'
    provisioning = 'provisioning'
    initializing = 'initializing'
    running = 'running'
    terminating = 'terminating'
    preempted = 'preempted'
    succeeded = 'succeeded'
    skipped = 'skipped'
    stopped = 'stopped'
    failed = 'failed'

    def __str__(self):
        if False:
            print('Hello World!')
        return self.name

    def is_end_state(self):
        if False:
            return 10
        if self is BeakerStatus.preempted:
            return True
        elif self is BeakerStatus.succeeded:
            return True
        elif self is BeakerStatus.skipped:
            return True
        elif self is BeakerStatus.stopped:
            return True
        elif self is BeakerStatus.failed:
            return True
        else:
            return False

class BeakerWrapper:

    def get_status(self, experiment_id: str) -> BeakerStatus:
        if False:
            for i in range(10):
                print('nop')
        command = ['beaker', 'experiment', 'inspect', experiment_id]
        experiment_json = subprocess.check_output(command)
        experiment_data = json.loads(experiment_json)
        assert len(experiment_data) == 1, 'Experiment not created with run_with_beaker.py'
        assert len(experiment_data[0]['nodes']) == 1, 'Experiment not created with run_with_beaker.py'
        status = BeakerStatus(experiment_data[0]['nodes'][0]['status'])
        time.sleep(BEAKER_QUERY_INTERVAL_SECONDS)
        return status

    def resume(self, experiment_id: str) -> str:
        if False:
            print('Hello World!')
        command = ['beaker', 'experiment', 'resume', f'--experiment-name={experiment_id}']
        time.sleep(BEAKER_QUERY_INTERVAL_SECONDS)
        return subprocess.check_output(command, universal_newlines=True).strip()

def create_table(connection: Connection) -> None:
    if False:
        for i in range(10):
            print('nop')
    cursor = connection.cursor()
    create_table_statement = '\n    CREATE TABLE active_experiments\n    (experiment_id TEXT PRIMARY KEY, original_id TEXT, max_resumes INTEGER, current_resume INTEGER)\n    '
    cursor.execute(create_table_statement)
    connection.commit()

def start_autoresume(connection: Connection, experiment_id: str, max_resumes: int) -> None:
    if False:
        return 10
    cursor = connection.cursor()
    cursor.execute('INSERT INTO active_experiments VALUES (?, ?, ?, ?)', (experiment_id, experiment_id, max_resumes, 0))
    connection.commit()

def stop_autoresume(connection: Connection, experiment_id: str) -> None:
    if False:
        i = 10
        return i + 15
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM active_experiments WHERE experiment_id = ?', (experiment_id,))
    result = cursor.fetchall()
    assert result, f'Experiment {experiment_id} not found!'
    cursor.execute('DELETE FROM active_experiments WHERE experiment_id = ?', (experiment_id,))
    connection.commit()

def resume(connection: Connection, beaker: BeakerWrapper) -> None:
    if False:
        print('Hello World!')
    logger.info('Checking if resumes are needed.')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM active_experiments')
    experiments = cursor.fetchall()
    for experiment_row in experiments:
        (experiment_id, original_id, max_resumes, current_resume) = experiment_row
        status = beaker.get_status(experiment_id)
        if status.is_end_state():
            stop_autoresume(connection, experiment_id)
            if status is BeakerStatus.preempted:
                if current_resume >= max_resumes:
                    logger.info(f'Experiment {experiment_id} preempted too many times ({max_resumes}). Original experiment: {original_id}')
                else:
                    new_experiment_id = beaker.resume(experiment_id)
                    logger.info(f'Experiment {experiment_id} preempted ({current_resume}/{max_resumes}). Resuming as: {new_experiment_id} Original experiment: {original_id}')
                    cursor.execute('INSERT INTO active_experiments VALUES (?, ?, ?, ?)', (new_experiment_id, original_id, max_resumes, current_resume + 1))
                    connection.commit()
            else:
                logger.info(f'Experiment {experiment_id} completed with status: {status}. Original experiment: {original_id}')

class Action(Enum):
    start = 'start'
    stop = 'stop'
    resume = 'resume'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name

def main(args) -> None:
    if False:
        for i in range(10):
            print('nop')
    time.sleep(random.randint(0, args.random_delay_seconds))
    db_path = f'{dot_allennlp_dir}/resume.db'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='active_experiments'")
    tables = cursor.fetchall()
    if not tables:
        create_table(connection)
    crontab_l_result = subprocess.run(['crontab', '-l'], universal_newlines=True, stdout=PIPE, stderr=PIPE)
    if crontab_l_result.returncode == 0:
        current_crontab = crontab_l_result.stdout
    else:
        assert 'no crontab' in crontab_l_result.stderr, f'crontab failed: {crontab_l_result.stderr}'
        current_crontab = ''
    full_path = os.path.abspath(__file__)
    if full_path not in current_crontab:
        cron_line = f"*/10 * * * * bash -c 'export PATH={os.environ['PATH']}; python3 {full_path} --action=resume --random-delay-seconds=60'\n"
        new_crontab = current_crontab + cron_line
        subprocess.run(['crontab', '-'], input=new_crontab, encoding='utf-8')
    if args.action is Action.start:
        assert args.experiment_id
        start_autoresume(connection, args.experiment_id, args.max_resumes)
    elif args.action is Action.stop:
        assert args.experiment_id
        stop_autoresume(connection, args.experiment_id)
    elif args.action is Action.resume:
        beaker = BeakerWrapper()
        resume(connection, beaker)
    else:
        raise Exception(f'Unaccounted for action {args.action}')
    connection.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=Action, choices=list(Action), required=True)
    parser.add_argument('--experiment-id', type=str)
    parser.add_argument('--max-resumes', type=int, default=10)
    parser.add_argument('--random-delay-seconds', type=int, default=0)
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        logger.exception('Fatal error')
        raise