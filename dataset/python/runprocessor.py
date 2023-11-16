import logging
import os
import signal
import time
import typing
from argparse import ArgumentParser
from datetime import timedelta
from importlib import reload

from django.core.management import BaseCommand
from django.utils import timezone

from task_processor import tasks
from task_processor.task_registry import registered_tasks
from task_processor.thread_monitoring import (
    clear_unhealthy_threads,
    write_unhealthy_threads,
)
from task_processor.threads import TaskRunner

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The `register_recurring_task` decorator is executed in the global scope,
        # meaning it will run whenever the code is executed, including when running
        # `manage.py` commands such as `migrate` or `showmigrations`. However, this
        # decorator saves the task to the database, which is not desired unless it is
        # being run by processor, and to indicate the we set this `RUN_BY_PROCESSOR`
        # environment variable.
        os.environ["RUN_BY_PROCESSOR"] = "True"

        # Since the tasks module is loaded by the ready method in TaskProcessorConfig
        # which is run before the command is initialised, we need to reload the internal
        # tasks module here to make sure recurring tasks are registered correctly.
        reload(tasks)

        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)

        self._threads: typing.List[TaskRunner] = []
        self._monitor_threads = True

    def add_arguments(self, parser: ArgumentParser):
        parser.add_argument(
            "--numthreads",
            type=int,
            help="Number of worker threads to run.",
            default=5,
        )
        parser.add_argument(
            "--sleepintervalms",
            type=int,
            help="Number of millis each worker waits before checking for new tasks",
            default=2000,
        )
        parser.add_argument(
            "--graceperiodms",
            type=int,
            help="Number of millis before running task is considered 'stuck'.",
            default=3000,
        )
        parser.add_argument(
            "--queuepopsize",
            type=int,
            help="Number of tasks each worker will pop from the queue on each cycle.",
            default=10,
        )

    def handle(self, *args, **options):
        num_threads = options["numthreads"]
        sleep_interval_ms = options["sleepintervalms"]
        grace_period_ms = options["graceperiodms"]
        queue_pop_size = options["queuepopsize"]

        self._threads.extend(
            [
                TaskRunner(
                    sleep_interval_millis=sleep_interval_ms,
                    queue_pop_size=queue_pop_size,
                )
                for _ in range(num_threads)
            ]
        )

        logger.info(
            "Processor starting. Registered tasks are: %s",
            list(registered_tasks.keys()),
        )

        for thread in self._threads:
            thread.start()

        clear_unhealthy_threads()
        while self._monitor_threads:
            time.sleep(1)
            unhealthy_threads = self._get_unhealthy_threads(
                ms_before_unhealthy=grace_period_ms + sleep_interval_ms
            )
            if unhealthy_threads:
                write_unhealthy_threads(unhealthy_threads)

        [t.join() for t in self._threads]

    def _exit_gracefully(self, *args):
        self._monitor_threads = False
        for t in self._threads:
            t.stop()

    def _get_unhealthy_threads(
        self, ms_before_unhealthy: int
    ) -> typing.List[TaskRunner]:
        unhealthy_threads = []
        healthy_threshold = timezone.now() - timedelta(milliseconds=ms_before_unhealthy)

        for thread in self._threads:
            if (
                not thread.is_alive()
                or not thread.last_checked_for_tasks
                or thread.last_checked_for_tasks < healthy_threshold
            ):
                unhealthy_threads.append(thread)
        return unhealthy_threads
