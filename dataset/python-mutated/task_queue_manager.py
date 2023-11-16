from __future__ import annotations
import os
import sys
import tempfile
import threading
import time
import typing as t
import multiprocessing.queues
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError
from ansible.executor.play_iterator import PlayIterator
from ansible.executor.stats import AggregateStats
from ansible.executor.task_result import TaskResult
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.playbook.play_context import PlayContext
from ansible.playbook.task import Task
from ansible.plugins.loader import callback_loader, strategy_loader, module_loader
from ansible.plugins.callback import CallbackBase
from ansible.template import Templar
from ansible.vars.hostvars import HostVars
from ansible.vars.reserved import warn_if_reserved
from ansible.utils.display import Display
from ansible.utils.lock import lock_decorator
from ansible.utils.multiprocessing import context as multiprocessing_context
from dataclasses import dataclass
__all__ = ['TaskQueueManager']
display = Display()

class CallbackSend:

    def __init__(self, method_name, *args, **kwargs):
        if False:
            print('Hello World!')
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

class DisplaySend:

    def __init__(self, method, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.method = method
        self.args = args
        self.kwargs = kwargs

@dataclass
class PromptSend:
    worker_id: int
    prompt: str
    private: bool = True
    seconds: int = None
    interrupt_input: t.Iterable[bytes] = None
    complete_input: t.Iterable[bytes] = None

class FinalQueue(multiprocessing.queues.SimpleQueue):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['ctx'] = multiprocessing_context
        super().__init__(*args, **kwargs)

    def send_callback(self, method_name, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.put(CallbackSend(method_name, *args, **kwargs))

    def send_task_result(self, *args, **kwargs):
        if False:
            return 10
        if isinstance(args[0], TaskResult):
            tr = args[0]
        else:
            tr = TaskResult(*args, **kwargs)
        self.put(tr)

    def send_display(self, method, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.put(DisplaySend(method, *args, **kwargs))

    def send_prompt(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.put(PromptSend(**kwargs))

class AnsibleEndPlay(Exception):

    def __init__(self, result):
        if False:
            print('Hello World!')
        self.result = result

class TaskQueueManager:
    """
    This class handles the multiprocessing requirements of Ansible by
    creating a pool of worker forks, a result handler fork, and a
    manager object with shared datastructures/queues for coordinating
    work between all processes.

    The queue manager is responsible for loading the play strategy plugin,
    which dispatches the Play's tasks to hosts.
    """
    RUN_OK = 0
    RUN_ERROR = 1
    RUN_FAILED_HOSTS = 2
    RUN_UNREACHABLE_HOSTS = 4
    RUN_FAILED_BREAK_PLAY = 8
    RUN_UNKNOWN_ERROR = 255

    def __init__(self, inventory, variable_manager, loader, passwords, stdout_callback=None, run_additional_callbacks=True, run_tree=False, forks=None):
        if False:
            for i in range(10):
                print('nop')
        self._inventory = inventory
        self._variable_manager = variable_manager
        self._loader = loader
        self._stats = AggregateStats()
        self.passwords = passwords
        self._stdout_callback = stdout_callback
        self._run_additional_callbacks = run_additional_callbacks
        self._run_tree = run_tree
        self._forks = forks or 5
        self._callbacks_loaded = False
        self._callback_plugins = []
        self._start_at_done = False
        if context.CLIARGS.get('module_path', False):
            for path in context.CLIARGS['module_path']:
                if path:
                    module_loader.add_directory(path)
        self._terminated = False
        self._failed_hosts = dict()
        self._unreachable_hosts = dict()
        try:
            self._final_q = FinalQueue()
        except OSError as e:
            raise AnsibleError('Unable to use multiprocessing, this is normally caused by lack of access to /dev/shm: %s' % to_native(e))
        self._callback_lock = threading.Lock()
        self._connection_lockfile = tempfile.TemporaryFile()

    def _initialize_processes(self, num):
        if False:
            while True:
                i = 10
        self._workers = []
        for i in range(num):
            self._workers.append(None)

    def load_callbacks(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Loads all available callbacks, with the exception of those which\n        utilize the CALLBACK_TYPE option. When CALLBACK_TYPE is set to 'stdout',\n        only one such callback plugin will be loaded.\n        "
        if self._callbacks_loaded:
            return
        stdout_callback_loaded = False
        if self._stdout_callback is None:
            self._stdout_callback = C.DEFAULT_STDOUT_CALLBACK
        if isinstance(self._stdout_callback, CallbackBase):
            stdout_callback_loaded = True
        elif isinstance(self._stdout_callback, string_types):
            if self._stdout_callback not in callback_loader:
                raise AnsibleError('Invalid callback for stdout specified: %s' % self._stdout_callback)
            else:
                self._stdout_callback = callback_loader.get(self._stdout_callback)
                self._stdout_callback.set_options()
                stdout_callback_loaded = True
        else:
            raise AnsibleError('callback must be an instance of CallbackBase or the name of a callback plugin')
        callback_list = list(callback_loader.all(class_only=True))
        for c in C.CALLBACKS_ENABLED:
            plugin = callback_loader.get(c, class_only=True)
            if plugin:
                if plugin not in callback_list:
                    callback_list.append(plugin)
            else:
                display.warning("Skipping callback plugin '%s', unable to load" % c)
        for callback_plugin in callback_list:
            callback_type = getattr(callback_plugin, 'CALLBACK_TYPE', '')
            callback_needs_enabled = getattr(callback_plugin, 'CALLBACK_NEEDS_ENABLED', getattr(callback_plugin, 'CALLBACK_NEEDS_WHITELIST', False))
            cnames = getattr(callback_plugin, '_redirected_names', [])
            if cnames:
                callback_name = cnames[0]
            else:
                (callback_name, ext) = os.path.splitext(os.path.basename(callback_plugin._original_path))
            display.vvvvv("Attempting to use '%s' callback." % callback_name)
            if callback_type == 'stdout':
                if callback_name != self._stdout_callback or stdout_callback_loaded:
                    display.vv("Skipping callback '%s', as we already have a stdout callback." % callback_name)
                    continue
                stdout_callback_loaded = True
            elif callback_name == 'tree' and self._run_tree:
                pass
            elif not self._run_additional_callbacks or (callback_needs_enabled and (C.CALLBACKS_ENABLED is None or callback_name not in C.CALLBACKS_ENABLED)):
                continue
            try:
                callback_obj = callback_plugin()
                if callback_obj:
                    if callback_obj not in self._callback_plugins:
                        callback_obj.set_options()
                        self._callback_plugins.append(callback_obj)
                    else:
                        display.vv("Skipping callback '%s', already loaded as '%s'." % (callback_plugin, callback_name))
                else:
                    display.warning("Skipping callback '%s', as it does not create a valid plugin instance." % callback_name)
                    continue
            except Exception as e:
                display.warning("Skipping callback '%s', unable to load due to: %s" % (callback_name, to_native(e)))
                continue
        self._callbacks_loaded = True

    def run(self, play):
        if False:
            print('Hello World!')
        '\n        Iterates over the roles/tasks in a play, using the given (or default)\n        strategy for queueing tasks. The default is the linear strategy, which\n        operates like classic Ansible by keeping all hosts in lock-step with\n        a given task (meaning no hosts move on to the next task until all hosts\n        are done with the current task).\n        '
        if not self._callbacks_loaded:
            self.load_callbacks()
        all_vars = self._variable_manager.get_vars(play=play)
        templar = Templar(loader=self._loader, variables=all_vars)
        warn_if_reserved(all_vars, templar.environment.globals.keys())
        new_play = play.copy()
        new_play.post_validate(templar)
        new_play.handlers = new_play.compile_roles_handlers() + new_play.handlers
        self.hostvars = HostVars(inventory=self._inventory, variable_manager=self._variable_manager, loader=self._loader)
        play_context = PlayContext(new_play, self.passwords, self._connection_lockfile.fileno())
        if self._stdout_callback and hasattr(self._stdout_callback, 'set_play_context'):
            self._stdout_callback.set_play_context(play_context)
        for callback_plugin in self._callback_plugins:
            if hasattr(callback_plugin, 'set_play_context'):
                callback_plugin.set_play_context(play_context)
        self.send_callback('v2_playbook_on_play_start', new_play)
        iterator = PlayIterator(inventory=self._inventory, play=new_play, play_context=play_context, variable_manager=self._variable_manager, all_vars=all_vars, start_at_done=self._start_at_done)
        self._initialize_processes(min(self._forks, iterator.batch_size))
        strategy = strategy_loader.get(new_play.strategy, self)
        if strategy is None:
            raise AnsibleError('Invalid play strategy specified: %s' % new_play.strategy, obj=play._ds)
        for host_name in self._failed_hosts.keys():
            host = self._inventory.get_host(host_name)
            iterator.mark_host_failed(host)
        for host_name in self._unreachable_hosts.keys():
            iterator._play._removed_hosts.append(host_name)
        self.clear_failed_hosts()
        if context.CLIARGS.get('start_at_task') is not None and play_context.start_at_task is None:
            self._start_at_done = True
        try:
            play_return = strategy.run(iterator, play_context)
        finally:
            strategy.cleanup()
            self._cleanup_processes()
        for host_name in iterator.get_failed_hosts():
            self._failed_hosts[host_name] = True
        if iterator.end_play:
            raise AnsibleEndPlay(play_return)
        return play_return

    def cleanup(self):
        if False:
            i = 10
            return i + 15
        display.debug('RUNNING CLEANUP')
        self.terminate()
        self._final_q.close()
        self._cleanup_processes()
        sys.stdout.flush()
        sys.stderr.flush()

    def _cleanup_processes(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '_workers'):
            for attempts_remaining in range(C.WORKER_SHUTDOWN_POLL_COUNT - 1, -1, -1):
                if not any((worker_prc and worker_prc.is_alive() for worker_prc in self._workers)):
                    break
                if attempts_remaining:
                    time.sleep(C.WORKER_SHUTDOWN_POLL_DELAY)
                else:
                    display.warning('One or more worker processes are still running and will be terminated.')
            for worker_prc in self._workers:
                if worker_prc and worker_prc.is_alive():
                    try:
                        worker_prc.terminate()
                    except AttributeError:
                        pass

    def clear_failed_hosts(self):
        if False:
            for i in range(10):
                print('nop')
        self._failed_hosts = dict()

    def get_inventory(self):
        if False:
            i = 10
            return i + 15
        return self._inventory

    def get_variable_manager(self):
        if False:
            for i in range(10):
                print('nop')
        return self._variable_manager

    def get_loader(self):
        if False:
            return 10
        return self._loader

    def get_workers(self):
        if False:
            i = 10
            return i + 15
        return self._workers[:]

    def terminate(self):
        if False:
            print('Hello World!')
        self._terminated = True

    def has_dead_workers(self):
        if False:
            print('Hello World!')
        defunct = False
        for x in self._workers:
            if getattr(x, 'exitcode', None):
                defunct = True
        return defunct

    @lock_decorator(attr='_callback_lock')
    def send_callback(self, method_name, *args, **kwargs):
        if False:
            return 10
        for callback_plugin in [self._stdout_callback] + self._callback_plugins:
            if getattr(callback_plugin, 'disabled', False):
                continue
            wants_implicit_tasks = getattr(callback_plugin, 'wants_implicit_tasks', False)
            methods = []
            for possible in [method_name, 'v2_on_any']:
                gotit = getattr(callback_plugin, possible, None)
                if gotit is None:
                    gotit = getattr(callback_plugin, possible.removeprefix('v2_'), None)
                if gotit is not None:
                    methods.append(gotit)
            new_args = []
            is_implicit_task = False
            for arg in args:
                if isinstance(arg, TaskResult):
                    new_args.append(arg.clean_copy())
                else:
                    new_args.append(arg)
                if isinstance(arg, Task) and arg.implicit:
                    is_implicit_task = True
            if is_implicit_task and (not wants_implicit_tasks):
                continue
            for method in methods:
                try:
                    method(*new_args, **kwargs)
                except Exception as e:
                    display.warning(u'Failure using method (%s) in callback plugin (%s): %s' % (to_text(method_name), to_text(callback_plugin), to_text(e)))
                    from traceback import format_tb
                    from sys import exc_info
                    display.vvv('Callback Exception: \n' + ' '.join(format_tb(exc_info()[2])))