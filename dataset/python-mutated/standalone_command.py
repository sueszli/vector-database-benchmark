from __future__ import annotations
import logging
import os
import socket
import subprocess
import threading
import time
from collections import deque
from typing import TYPE_CHECKING
from termcolor import colored
from airflow.configuration import conf
from airflow.executors import executor_constants
from airflow.executors.executor_loader import ExecutorLoader
from airflow.jobs.job import most_recent_job
from airflow.jobs.scheduler_job_runner import SchedulerJobRunner
from airflow.jobs.triggerer_job_runner import TriggererJobRunner
from airflow.utils import db
from airflow.utils.providers_configuration_loader import providers_configuration_loaded
if TYPE_CHECKING:
    from airflow.jobs.base_job_runner import BaseJobRunner

class StandaloneCommand:
    """
    Runs all components of Airflow under a single parent process.

    Useful for local development.
    """

    @classmethod
    def entrypoint(cls, args):
        if False:
            return 10
        'CLI entrypoint, called by the main CLI system.'
        StandaloneCommand().run()

    def __init__(self):
        if False:
            print('Hello World!')
        self.subcommands = {}
        self.output_queue = deque()
        self.user_info = {}
        self.ready_time = None
        self.ready_delay = 3

    @providers_configuration_loaded
    def run(self):
        if False:
            while True:
                i = 10
        self.print_output('standalone', 'Starting Airflow Standalone')
        logging.getLogger('').setLevel(logging.WARNING)
        env = self.calculate_env()
        self.initialize_database()
        self.subcommands['scheduler'] = SubCommand(self, name='scheduler', command=['scheduler'], env=env)
        self.subcommands['webserver'] = SubCommand(self, name='webserver', command=['webserver'], env=env)
        self.subcommands['triggerer'] = SubCommand(self, name='triggerer', command=['triggerer'], env=env)
        self.web_server_port = conf.getint('webserver', 'WEB_SERVER_PORT', fallback=8080)
        for command in self.subcommands.values():
            command.start()
        shown_ready = False
        try:
            while True:
                self.update_output()
                if not self.ready_time and self.is_ready():
                    self.ready_time = time.monotonic()
                if not shown_ready and self.ready_time and (time.monotonic() - self.ready_time > self.ready_delay):
                    self.print_ready()
                    shown_ready = True
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        self.print_output('standalone', 'Shutting down components')
        for command in self.subcommands.values():
            command.stop()
        for command in self.subcommands.values():
            command.join()
        self.print_output('standalone', 'Complete')

    def update_output(self):
        if False:
            i = 10
            return i + 15
        'Drains the output queue and prints its contents to the screen.'
        while self.output_queue:
            (name, line) = self.output_queue.popleft()
            line_str = line.decode('utf8').strip()
            self.print_output(name, line_str)

    def print_output(self, name: str, output):
        if False:
            while True:
                i = 10
        '\n        Print an output line with name and colouring.\n\n        You can pass multiple lines to output if you wish; it will be split for you.\n        '
        color = {'webserver': 'green', 'scheduler': 'blue', 'triggerer': 'cyan', 'standalone': 'white'}.get(name, 'white')
        colorised_name = colored(f'{name:10}', color)
        for line in output.splitlines():
            print(f'{colorised_name} | {line.strip()}')

    def print_error(self, name: str, output):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print an error message to the console.\n\n        This is the same as print_output but with the text red\n        '
        self.print_output(name, colored(output, 'red'))

    def calculate_env(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Works out the environment variables needed to run subprocesses.\n\n        We override some settings as part of being standalone.\n        '
        env = dict(os.environ)
        (executor_class, _) = ExecutorLoader.import_default_executor_cls()
        if not executor_class.is_local:
            if 'sqlite' in conf.get('database', 'sql_alchemy_conn'):
                self.print_output('standalone', 'Forcing executor to SequentialExecutor')
                env['AIRFLOW__CORE__EXECUTOR'] = executor_constants.SEQUENTIAL_EXECUTOR
            else:
                self.print_output('standalone', 'Forcing executor to LocalExecutor')
                env['AIRFLOW__CORE__EXECUTOR'] = executor_constants.LOCAL_EXECUTOR
        return env

    def initialize_database(self):
        if False:
            while True:
                i = 10
        'Make sure all the tables are created.'
        self.print_output('standalone', 'Checking database is initialized')
        db.initdb()
        self.print_output('standalone', 'Database ready')
        from airflow.auth.managers.fab.cli_commands.utils import get_application_builder
        with get_application_builder() as appbuilder:
            (user_name, password) = appbuilder.sm.create_admin_standalone()
        self.user_info = {'username': user_name, 'password': password}

    def is_ready(self):
        if False:
            i = 10
            return i + 15
        "\n        Detect when all Airflow components are ready to serve.\n\n        For now, it's simply time-based.\n        "
        return self.port_open(self.web_server_port) and self.job_running(SchedulerJobRunner) and self.job_running(TriggererJobRunner)

    def port_open(self, port):
        if False:
            print('Hello World!')
        '\n        Check if the given port is listening on the local machine.\n\n        Used to tell if webserver is alive.\n        '
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect(('127.0.0.1', port))
            sock.close()
        except (OSError, ValueError):
            return False
        return True

    def job_running(self, job_runner_class: type[BaseJobRunner]):
        if False:
            i = 10
            return i + 15
        '\n        Check if the given job name is running and heartbeating correctly.\n\n        Used to tell if scheduler is alive.\n        '
        recent = most_recent_job(job_runner_class.job_type)
        if not recent:
            return False
        return recent.is_alive()

    def print_ready(self):
        if False:
            return 10
        '\n        Print the banner shown when Airflow is ready to go.\n\n        Include with login details.\n        '
        self.print_output('standalone', '')
        self.print_output('standalone', 'Airflow is ready')
        if self.user_info['password']:
            self.print_output('standalone', f"Login with username: {self.user_info['username']}  password: {self.user_info['password']}")
        self.print_output('standalone', 'Airflow Standalone is for development purposes only. Do not use this in production!')
        self.print_output('standalone', '')

class SubCommand(threading.Thread):
    """
    Execute a subcommand on another thread.

    Thread that launches a process and then streams its output back to the main
    command. We use threads to avoid using select() and raw filehandles, and the
    complex logic that brings doing line buffering.
    """

    def __init__(self, parent, name: str, command: list[str], env: dict[str, str]):
        if False:
            while True:
                i = 10
        super().__init__()
        self.parent = parent
        self.name = name
        self.command = command
        self.env = env

    def run(self):
        if False:
            return 10
        'Run the actual process and captures it output to a queue.'
        self.process = subprocess.Popen(['airflow', *self.command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self.env)
        for line in self.process.stdout:
            self.parent.output_queue.append((self.name, line))

    def stop(self):
        if False:
            return 10
        'Call to stop this process (and thus this thread).'
        self.process.terminate()
standalone = StandaloneCommand.entrypoint