import asyncio
import concurrent.futures
import functools
import logging
from coalib.core.DependencyTracker import DependencyTracker
from coalib.core.Graphs import traverse_graph
from coalib.core.PersistentHash import persistent_hash
from coalib.misc.Compatibility import run_coroutine_threadsafe

def group(iterable, key=lambda x: x):
    if False:
        print('Hello World!')
    '\n    Groups elements (out-of-order) together in the given iterable.\n\n    Supports non-hashable keys by comparing keys with ``==``.\n\n    Accessing the groups is supported using the iterator as follows:\n\n    >>> for key, elements in group([1, 3, 7, 1, 2, 1, 2]):\n    ...     print(key, list(elements))\n    1 [1, 1, 1]\n    3 [3]\n    7 [7]\n    2 [2, 2]\n\n    You can control how elements are grouped by using the ``key`` parameter. It\n    takes a function with a single parameter and maps to the group.\n\n    >>> data = [(1, 2), (3, 4), (1, 9), (2, 10), (1, 11), (7, 2), (10, 2),\n    ...         (2, 1), (3, 7), (4, 5)]\n    >>> for key, elements in group(data, key=sum):\n    ...     print(key, list(elements))\n    3 [(1, 2), (2, 1)]\n    7 [(3, 4)]\n    10 [(1, 9), (3, 7)]\n    12 [(2, 10), (1, 11), (10, 2)]\n    9 [(7, 2), (4, 5)]\n\n    :param iterable:\n        The iterable to group elements in.\n    :param key:\n        The key-function mapping an element to its group.\n    :return:\n        An iterable yielding tuples with ``key, elements``, where ``elements``\n        is also an iterable yielding the elements grouped under ``key``.\n    '
    keys = []
    elements = []
    for element in iterable:
        k = key(element)
        try:
            position = keys.index(k)
            element_list = elements[position]
        except ValueError:
            keys.append(k)
            element_list = []
            elements.append(element_list)
        element_list.append(element)
    return zip(keys, elements)

def initialize_dependencies(bears):
    if False:
        for i in range(10):
            print('nop')
    '\n    Initializes and returns a ``DependencyTracker`` instance together with a\n    set of bears ready for scheduling.\n\n    This function acquires, processes and registers bear dependencies\n    accordingly using a consumer-based system, where each dependency bear has\n    only a single instance per section and file-dictionary.\n\n    The bears set returned accounts for bears that have dependencies and\n    excludes them accordingly. Dependency bears that have themselves no further\n    dependencies are included so the dependency chain can be processed\n    correctly.\n\n    :param bears:\n        The set of instantiated bears to run that serve as an entry-point.\n    :return:\n        A tuple with ``(dependency_tracker, bears_to_schedule)``.\n    '
    bears = set(bears)
    dependency_tracker = DependencyTracker()
    grouping = group(bears, key=lambda bear: (bear.section, bear.file_dict))
    for ((section, file_dict), bears_per_section) in grouping:
        bears_per_section = list(bears_per_section)
        type_to_instance_map = {}
        for bear in bears_per_section:
            type_to_instance_map[bear] = bear
            type_to_instance_map[type(bear)] = bear

        def get_successive_nodes_and_track(bear):
            if False:
                print('Hello World!')
            for dependency_bear_type in bear.BEAR_DEPS:
                if dependency_bear_type not in type_to_instance_map:
                    dependency_bear = dependency_bear_type(section, file_dict)
                    type_to_instance_map[dependency_bear_type] = dependency_bear
                dependency_tracker.add(type_to_instance_map[dependency_bear_type], bear)
            return (type_to_instance_map[dependency_bear_type] for dependency_bear_type in bear.BEAR_DEPS)
        traverse_graph(bears_per_section, get_successive_nodes_and_track)
    bears -= {bear for bear in bears if dependency_tracker.get_dependencies(bear)}
    for dependency in dependency_tracker.dependencies:
        if not dependency_tracker.get_dependencies(dependency):
            bears.add(dependency)
    return (dependency_tracker, bears)

class Session:
    """
    Maintains a session for a coala execution. For each session, there are set
    of bears to run along with a callback function, which is called when
    results are available.

    Dependencies of bears (provided via ``bear.BEAR_DEPS``) are automatically
    handled. If BearA requires BearB as dependency, then on running BearA,
    first BearB will be executed, followed by BearA.
    """

    def __init__(self, bears, result_callback, cache=None, executor=None):
        if False:
            print('Hello World!')
        "\n        :param bears:\n            The bear instances to run.\n        :param result_callback:\n            A callback function which is called when results are available.\n            Must have following signature::\n\n                def result_callback(result):\n                    pass\n\n            Only those results are passed for bears that were explicitly\n            requested via the ``bears`` parameter, implicit dependency results\n            do not call the callback.\n        :param cache:\n            A cache bears can use to speed up runs. If ``None``, no cache will\n            be used.\n\n            The cache stores the results that were returned last time from the\n            parameters passed to ``execute_task`` in bears. If the parameters\n            to ``execute_task`` are the same from a previous run, the cache\n            will be queried instead of executing ``execute_task``.\n\n            The cache has to be a dictionary-like object, that maps bear types\n            to respective cache-tables. The cache-tables itself are\n            dictionary-like objects that map hash-values (generated by\n            ``PersistentHash.persistent_hash`` from the task objects) to actual\n            bear results. When bears are about to be scheduled, the core\n            performs a cache-lookup. If there's a hit, the results stored in\n            the cache are returned and the task won't be scheduled. In case of\n            a miss, ``execute_task`` is called normally in the executor.\n        :param executor:\n            Custom executor used to run the bears. If ``None``, a\n            ``ProcessPoolExecutor`` is used using as many processes as cores\n            available on the system. Note that a passed custom executor is\n            closed after the core has finished.\n        "
        self.bears = bears
        self.result_callback = result_callback
        self.cache = cache
        self.event_loop = asyncio.SelectorEventLoop()
        self.executor = concurrent.futures.ProcessPoolExecutor() if executor is None else executor
        self.running_futures = {}
        (self.dependency_tracker, self.bears_to_schedule) = initialize_dependencies(self.bears)

    def run(self):
        if False:
            print('Hello World!')
        '\n        Runs the coala session.\n        '
        try:
            if self.bears:
                self._schedule_bears(self.bears_to_schedule)
                try:
                    self.event_loop.run_forever()
                finally:
                    self.event_loop.close()
        finally:
            self.executor.shutdown()

    def _schedule_bears(self, bears):
        if False:
            print('Hello World!')
        '\n        Schedules the tasks of bears.\n\n        :param bears:\n            A list of bear instances to be scheduled onto the process pool.\n        '
        bears_without_tasks = []
        for bear in bears:
            if self.dependency_tracker.get_dependencies(bear):
                logging.warning(f'Dependencies for {bear!r} not yet resolved, holding back. This should not happen, the dependency tracking system should be smarter. Please report this to the developers.')
            else:
                futures = set()
                for task in bear.generate_tasks():
                    (bear_args, bear_kwargs) = task
                    if self.cache is None:
                        future = self.event_loop.run_in_executor(self.executor, bear.execute_task, bear_args, bear_kwargs)
                    else:
                        future = self.event_loop.run_in_executor(None, self._execute_task_with_cache, bear, task)
                    futures.add(future)
                self.running_futures[bear] = futures
                if not futures:
                    logging.debug(f'{bear!r} scheduled no tasks.')
                    bears_without_tasks.append(bear)
                    continue
                for future in futures:
                    future.add_done_callback(functools.partial(self._finish_task, bear))
                logging.debug(f'Scheduled {bear!r} (tasks: {len(futures)})')
        for bear in bears_without_tasks:
            self._cleanup_bear(bear)

    def _cleanup_bear(self, bear):
        if False:
            i = 10
            return i + 15
        '\n        Cleans up state of an ongoing run for a bear.\n\n        - If the given bear has no running tasks left:\n          - Resolves its dependencies.\n          - Schedules dependant bears.\n          - Removes the bear from the ``running_tasks`` dict.\n        - Checks whether there are any remaining tasks, and quits the event loop\n          accordingly if none are left.\n\n        :param bear:\n            The bear to clean up state for.\n        '
        if not self.running_futures[bear]:
            resolved_bears = self.dependency_tracker.resolve(bear)
            if resolved_bears:
                self._schedule_bears(resolved_bears)
            del self.running_futures[bear]
        if not self.running_futures:
            resolved = self.dependency_tracker.are_dependencies_resolved
            if not resolved:
                joined = ', '.join((repr(dependant) + ' depends on ' + repr(dependency) for (dependency, dependant) in self.dependency_tracker))
                logging.warning(f'Core finished with run, but it seems some dependencies were unresolved: {joined}. Ignoring them, but this is a bug, please report it to the developers.')
            self.event_loop.stop()

    def _execute_task_with_cache(self, bear, task):
        if False:
            i = 10
            return i + 15
        if type(bear) not in self.cache:
            bear_cache = {}
            self.cache[type(bear)] = bear_cache
        else:
            bear_cache = self.cache[type(bear)]
        fingerprint = persistent_hash(task)
        if fingerprint in bear_cache:
            results = bear_cache[fingerprint]
        else:
            (bear_args, bear_kwargs) = task
            future = run_coroutine_threadsafe(asyncio.wait_for(self.event_loop.run_in_executor(self.executor, bear.execute_task, bear_args, bear_kwargs), None, loop=self.event_loop), loop=self.event_loop)
            results = future.result()
            bear_cache[fingerprint] = results
        return results

    def _finish_task(self, bear, future):
        if False:
            while True:
                i = 10
        '\n        The callback for when a task of a bear completes. It is responsible for\n        checking if the bear completed its execution and the handling of the\n        result generated by the task. It also schedules new tasks if\n        dependencies get resolved.\n\n        :param bear:\n            The bear that the task belongs to.\n        :param future:\n            The future that completed.\n        '
        try:
            results = future.result()
            for dependant in self.dependency_tracker.get_dependants(bear):
                dependant.dependency_results[type(bear)] += results
        except Exception as ex:
            logging.error('An exception was thrown during bear execution.', exc_info=ex)
            results = None
            dependants = self.dependency_tracker.get_all_dependants(bear)
            for dependant in dependants:
                self.dependency_tracker.resolve(dependant)
            logging.debug('Following dependent bears were unscheduled: ' + ', '.join((repr(dependant) for dependant in dependants)))
        finally:
            self.running_futures[bear].remove(future)
            self._cleanup_bear(bear)
        if results is not None and bear in self.bears:
            for result in results:
                try:
                    self.result_callback(result)
                except Exception as ex:
                    logging.error('An exception was thrown during result-handling.', exc_info=ex)

def run(bears, result_callback, cache=None, executor=None):
    if False:
        return 10
    "\n    Initiates a session with the given parameters and runs it.\n\n    :param bears:\n        The bear instances to run.\n    :param result_callback:\n        A callback function which is called when results are available. Must\n        have following signature::\n\n            def result_callback(result):\n                pass\n\n        Only those results are passed for bears that were explicitly requested\n        via the ``bears`` parameter, implicit dependency results do not call\n        the callback.\n    :param cache:\n        A cache bears can use to speed up runs. If ``None``, no cache will be\n        used.\n\n        The cache stores the results that were returned last time from the\n        parameters passed to ``execute_task`` in bears. If the parameters\n        to ``execute_task`` are the same from a previous run, the cache\n        will be queried instead of executing ``execute_task``.\n\n        The cache has to be a dictionary-like object, that maps bear types\n        to respective cache-tables. The cache-tables itself are dictionary-like\n        objects that map hash-values (generated by\n        ``PersistentHash.persistent_hash`` from the task objects) to actual\n        bear results. When bears are about to be scheduled, the core performs\n        a cache-lookup. If there's a hit, the results stored in the cache\n        are returned and the task won't be scheduled. In case of a miss,\n        ``execute_task`` is called normally in the executor.\n    :param executor:\n        Custom executor used to run the bears. If ``None``, a\n        ``ProcessPoolExecutor`` is used using as many processes as cores\n        available on the system.\n    "
    Session(bears, result_callback, cache, executor).run()