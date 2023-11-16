"""An emulator that mocks the core.platform.taskqueue API. This emulator
models the third party library, Google Cloud Tasks.

This emulator is an extension of the emulator from this github page:
https://github.com/doitintl/Cloud-Tasks-In-Process-Emulator
"""
from __future__ import annotations
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
if TYPE_CHECKING:
    import datetime

class Task:
    """A mock for a Google Cloud Tasks task that is handled by execution using
    the cloud tasks emulator.
    """

    def __init__(self, queue_name: str, url: str, payload: Optional[Dict[str, Any]]=None, scheduled_for: Optional[float]=None, task_name: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Initialize a Task that can be executed by making a post request to\n        the given url with the correct data payload.\n\n        Args:\n            queue_name: str. The name of the queue to add the http task to.\n            url: str. URL of the handler function.\n            payload: dict(str : *). Payload to pass to the request. Defaults\n                to None if no payload is required.\n            scheduled_for: time|None. The time in which to execute the task,\n                relative to time.time().\n            task_name: str|None. Optional. The name of the task.\n        '
        self.payload = payload
        self.url = url
        self.scheduled_for = scheduled_for
        self.queue_name = queue_name
        self.task_name = task_name

class Emulator:
    """The emulator mocks the core.platform.taskqueue API. The queues in this
    emulator are priority queues: Elements are popped in the order of the time
    they are scheduled for and executed after the time for execution has
    been reached.

    This emulator exposes functionality that is used to provide a taskqueue API
    for both the App Engine development server and the backend unit tests.

    This emulator provides 2 types of functionality for task handling:
        1. The emulator will handle multiple priority queues containing tasks.
           One persistent thread is instantiated per priority queue and will
           constantly try to execute the next task prioritized by the
           'scheduled_for' attribute which determines when the task should be
           run.
        2. If automatic task handling is disabled, then the threads for each
           queue will not be created. Instead, the tasks will be added to the
           taskqueue by calling create_task() and tasks in an individual queue
           can be executed using process_and_flush_tasks().
    """

    def __init__(self, task_handler: Callable[..., Any], automatic_task_handling: bool=True) -> None:
        if False:
            return 10
        'Initializes the emulator with an empty task queue and the correct\n        task_handler callback.\n\n        Args:\n            task_handler: function. The function that will handle the task\n                execution.\n            automatic_task_handling: bool. Boolean value to determine whether\n                the emulator will handle tasks automatically via threads or\n                via user function calls as detailed in the docstring for this\n                emulator.\n        '
        self._lock = threading.Lock()
        self._task_handler = task_handler
        self._queues: Dict[str, List[Task]] = {}
        self.automatic_task_handling = automatic_task_handling
        self._queue_threads: Dict[str, threading.Thread] = {}

    def _process_queue(self, queue_name: str) -> None:
        if False:
            i = 10
            return i + 15
        'The callback function for each individual queue thread. Each queue\n        thread repeatedly queries the queue, pops tasks, and executes the tasks\n        that need to be executed.\n\n        Args:\n            queue_name: str. The name of the queue.\n        '
        while True:
            task = None
            with self._lock:
                queue = self._queues[queue_name]
                if queue:
                    peek = queue[0]
                    now = time.time()
                    assert peek.scheduled_for is not None
                    if peek.scheduled_for <= now:
                        task = queue.pop(0)
            if task:
                self._task_handler(url=task.url, payload=task.payload, queue_name=task.queue_name, task_name=task.task_name)
            time.sleep(0.01)

    def _launch_queue_thread(self, queue_name: str) -> None:
        if False:
            return 10
        'Launches a persistent thread for an individual queue in the\n        taskqueue.\n\n        Args:\n            queue_name: str. The name of the queue.\n        '
        new_thread = threading.Thread(target=self._process_queue, name='Thread-%s' % queue_name, args=[queue_name])
        new_thread.daemon = True
        self._queue_threads[queue_name] = new_thread
        new_thread.start()

    def _execute_tasks(self, task_list: List[Task]) -> None:
        if False:
            i = 10
            return i + 15
        'Executes all of the tasks in the task list using the task handler\n        callback.\n\n        Args:\n            task_list: list(Task). List of tasks to execute.\n        '
        for task in task_list:
            self._task_handler(url=task.url, payload=task.payload, queue_name=task.queue_name, task_name=task.task_name)

    def _total_enqueued_tasks(self) -> int:
        if False:
            return 10
        'Returns the total number of tasks across all of the queues in the\n        taskqueue.\n\n        Returns:\n            int. The total number of tasks in the taskqueue.\n        '
        return sum((len(q) for q in self._queues.values()))

    def create_task(self, queue_name: str, url: str, payload: Optional[Dict[str, Any]]=None, scheduled_for: Optional[datetime.datetime]=None, task_name: Optional[str]=None, retry: None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Creates a Task in the corresponding queue that will be executed when\n        the 'scheduled_for' time is reached. If the queue doesn't exist yet,\n        it will be created.\n\n        Args:\n            queue_name: str. The name of the queue to add the task to.\n            url: str. URL of the handler function.\n            payload: dict(str : *). Payload to pass to the request. Defaults\n                to None if no payload is required.\n            scheduled_for: datetime|None. The naive datetime object for the\n                time to execute the task. Pass in None for immediate execution.\n            task_name: str|None. Optional. The name of the task.\n            retry: None. The retry mechanism that should be used. Here we ignore\n                the value and it is not used for anything.\n        "
        scheduled_for_time = time.mktime(scheduled_for.timetuple()) if scheduled_for else time.time()
        with self._lock:
            if queue_name not in self._queues:
                self._queues[queue_name] = []
                if self.automatic_task_handling:
                    self._launch_queue_thread(queue_name)
            queue = self._queues[queue_name]
            task = Task(queue_name, url, payload, scheduled_for=scheduled_for_time, task_name=task_name)
            queue.append(task)
            k = lambda t: t.scheduled_for
            queue.sort(key=k)

    def get_number_of_tasks(self, queue_name: Optional[str]=None) -> int:
        if False:
            i = 10
            return i + 15
        'Returns the total number of tasks in a single queue if a queue name\n        is specified or the entire taskqueue if no queue name is specified.\n\n        Args:\n            queue_name: str|None. Name of the queue. Pass in None if no specific\n                queue is designated.\n\n        Returns:\n            int. The total number of tasks in a single queue or in the entire\n            taskqueue.\n        '
        if queue_name and queue_name in self._queues:
            return len(self._queues[queue_name])
        else:
            return self._total_enqueued_tasks()

    def process_and_flush_tasks(self, queue_name: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Executes all of the tasks in a single queue if a queue name is\n        specified or all of the tasks in the taskqueue if no queue name is\n        specified.\n\n        Args:\n            queue_name: str|None. Name of the queue. Pass in None if no specific\n                queue is designated.\n        '
        if queue_name and queue_name in self._queues:
            self._execute_tasks(self._queues[queue_name])
            self._queues[queue_name] = []
        else:
            for (queue, task_list) in self._queues.items():
                self._execute_tasks(task_list)
                self._queues[queue] = []

    def get_tasks(self, queue_name: Optional[str]=None) -> List[Task]:
        if False:
            while True:
                i = 10
        'Returns a list of the tasks in a single queue if a queue name is\n        specified or a list of all of the tasks in the taskqueue if no queue\n        name is specified.\n\n        Args:\n            queue_name: str|None. Name of the queue. Pass in None if no specific\n                queue is designated.\n\n        Returns:\n            list(Task). List of tasks in a single queue or in the entire\n            taskqueue.\n        '
        if queue_name:
            return self._queues[queue_name]
        else:
            tasks_list = []
            for items in self._queues.items():
                tasks_list.extend(items[1])
            return tasks_list