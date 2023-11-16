"""Server for mypy daemon mode.

This implements a daemon process which keeps useful state in memory
to enable fine-grained incremental reprocessing of changes.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import AbstractSet, Any, Callable, Final, List, Sequence, Tuple
from typing_extensions import TypeAlias as _TypeAlias
import mypy.build
import mypy.errors
import mypy.main
from mypy.dmypy_util import WriteToConn, receive, send
from mypy.find_sources import InvalidSourceList, create_source_list
from mypy.fscache import FileSystemCache
from mypy.fswatcher import FileData, FileSystemWatcher
from mypy.inspections import InspectionEngine
from mypy.ipc import IPCServer
from mypy.modulefinder import BuildSource, FindModuleCache, SearchPaths, compute_search_paths
from mypy.options import Options
from mypy.server.update import FineGrainedBuildManager, refresh_suppressed_submodules
from mypy.suggestions import SuggestionEngine, SuggestionFailure
from mypy.typestate import reset_global_state
from mypy.util import FancyFormatter, count_stats
from mypy.version import __version__
MEM_PROFILE: Final = False
if sys.platform == 'win32':
    from subprocess import STARTUPINFO

    def daemonize(options: Options, status_file: str, timeout: int | None=None, log_file: str | None=None) -> int:
        if False:
            while True:
                i = 10
        'Create the daemon process via "dmypy daemon" and pass options via command line\n\n        When creating the daemon grandchild, we create it in a new console, which is\n        started hidden. We cannot use DETACHED_PROCESS since it will cause console windows\n        to pop up when starting. See\n        https://github.com/python/cpython/pull/4150#issuecomment-340215696\n        for more on why we can\'t have nice things.\n\n        It also pickles the options to be unpickled by mypy.\n        '
        command = [sys.executable, '-m', 'mypy.dmypy', '--status-file', status_file, 'daemon']
        pickled_options = pickle.dumps(options.snapshot())
        command.append(f'--options-data="{base64.b64encode(pickled_options).decode()}"')
        if timeout:
            command.append(f'--timeout={timeout}')
        if log_file:
            command.append(f'--log-file={log_file}')
        info = STARTUPINFO()
        info.dwFlags = 1
        info.wShowWindow = 0
        try:
            subprocess.Popen(command, creationflags=16, startupinfo=info)
            return 0
        except subprocess.CalledProcessError as e:
            return e.returncode
else:

    def _daemonize_cb(func: Callable[[], None], log_file: str | None=None) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Arrange to call func() in a grandchild of the current process.\n\n        Return 0 for success, exit status for failure, negative if\n        subprocess killed by signal.\n        '
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid:
            (npid, sts) = os.waitpid(pid, 0)
            sig = sts & 255
            if sig:
                print('Child killed by signal', sig)
                return -sig
            sts = sts >> 8
            if sts:
                print('Child exit status', sts)
            return sts
        try:
            os.setsid()
            os.umask(23)
            devnull = os.open('/dev/null', os.O_RDWR)
            os.dup2(devnull, 0)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
            pid = os.fork()
            if pid:
                os._exit(0)
            if log_file:
                sys.stdout = sys.stderr = open(log_file, 'a', buffering=1)
                fd = sys.stdout.fileno()
                os.dup2(fd, 2)
                os.dup2(fd, 1)
            func()
        finally:
            os._exit(1)

    def daemonize(options: Options, status_file: str, timeout: int | None=None, log_file: str | None=None) -> int:
        if False:
            print('Hello World!')
        'Run the mypy daemon in a grandchild of the current process\n\n        Return 0 for success, exit status for failure, negative if\n        subprocess killed by signal.\n        '
        return _daemonize_cb(Server(options, status_file, timeout).serve, log_file)
CONNECTION_NAME: Final = 'dmypy'

def process_start_options(flags: list[str], allow_sources: bool) -> Options:
    if False:
        for i in range(10):
            print('nop')
    (_, options) = mypy.main.process_options(['-i'] + flags, require_targets=False, server_options=True)
    if options.report_dirs:
        print('dmypy: Ignoring report generation settings. Start/restart cannot generate reports.')
    if options.junit_xml:
        print('dmypy: Ignoring report generation settings. Start/restart does not support --junit-xml. Pass it to check/recheck instead')
        options.junit_xml = None
    if not options.incremental:
        sys.exit('dmypy: start/restart should not disable incremental mode')
    if options.follow_imports not in ('skip', 'error', 'normal'):
        sys.exit('dmypy: follow-imports=silent not supported')
    return options

def ignore_suppressed_imports(module: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Can we skip looking for newly unsuppressed imports to module?'
    return module.startswith('encodings.')
ModulePathPair: _TypeAlias = Tuple[str, str]
ModulePathPairs: _TypeAlias = List[ModulePathPair]
ChangesAndRemovals: _TypeAlias = Tuple[ModulePathPairs, ModulePathPairs]

class Server:

    def __init__(self, options: Options, status_file: str, timeout: int | None=None) -> None:
        if False:
            print('Hello World!')
        'Initialize the server with the desired mypy flags.'
        self.options = options
        self.options_snapshot = options.snapshot()
        self.timeout = timeout
        self.fine_grained_manager: FineGrainedBuildManager | None = None
        if os.path.isfile(status_file):
            os.unlink(status_file)
        self.fscache = FileSystemCache()
        options.raise_exceptions = True
        options.incremental = True
        options.fine_grained_incremental = True
        options.show_traceback = True
        if options.use_fine_grained_cache:
            options.cache_fine_grained = True
        else:
            options.cache_dir = os.devnull
        options.local_partial_types = True
        self.status_file = status_file
        self.formatter = FancyFormatter(sys.stdout, sys.stderr, options.hide_error_codes)

    def _response_metadata(self) -> dict[str, str]:
        if False:
            print('Hello World!')
        py_version = f'{self.options.python_version[0]}_{self.options.python_version[1]}'
        return {'platform': self.options.platform, 'python_version': py_version}

    def serve(self) -> None:
        if False:
            while True:
                i = 10
        'Serve requests, synchronously (no thread or fork).'
        command = None
        server = IPCServer(CONNECTION_NAME, self.timeout)
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        try:
            with open(self.status_file, 'w') as f:
                json.dump({'pid': os.getpid(), 'connection_name': server.connection_name}, f)
                f.write('\n')
            while True:
                with server:
                    data = receive(server)
                    sys.stdout = WriteToConn(server, 'stdout')
                    sys.stderr = WriteToConn(server, 'stderr')
                    resp: dict[str, Any] = {}
                    if 'command' not in data:
                        resp = {'error': 'No command found in request'}
                    else:
                        command = data['command']
                        if not isinstance(command, str):
                            resp = {'error': 'Command is not a string'}
                        else:
                            command = data.pop('command')
                            try:
                                resp = self.run_command(command, data)
                            except Exception:
                                tb = traceback.format_exception(*sys.exc_info())
                                resp = {'error': 'Daemon crashed!\n' + ''.join(tb)}
                                resp.update(self._response_metadata())
                                resp['final'] = True
                                send(server, resp)
                                raise
                    resp['final'] = True
                    try:
                        resp.update(self._response_metadata())
                        send(server, resp)
                    except OSError:
                        pass
                    if command == 'stop':
                        reset_global_state()
                        sys.exit(0)
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            if command != 'stop':
                os.unlink(self.status_file)
            try:
                server.cleanup()
            except OSError:
                pass
            exc_info = sys.exc_info()
            if exc_info[0] and exc_info[0] is not SystemExit:
                traceback.print_exception(*exc_info)

    def run_command(self, command: str, data: dict[str, object]) -> dict[str, object]:
        if False:
            while True:
                i = 10
        'Run a specific command from the registry.'
        key = 'cmd_' + command
        method = getattr(self.__class__, key, None)
        if method is None:
            return {'error': f"Unrecognized command '{command}'"}
        else:
            if command not in {'check', 'recheck', 'run'}:
                del data['is_tty']
                del data['terminal_width']
            ret = method(self, **data)
            assert isinstance(ret, dict)
            return ret

    def cmd_status(self, fswatcher_dump_file: str | None=None) -> dict[str, object]:
        if False:
            for i in range(10):
                print('nop')
        'Return daemon status.'
        res: dict[str, object] = {}
        res.update(get_meminfo())
        if fswatcher_dump_file:
            data = self.fswatcher.dump_file_data() if hasattr(self, 'fswatcher') else {}
            s = json.dumps(data)
            with open(fswatcher_dump_file, 'w') as f:
                f.write(s)
        return res

    def cmd_stop(self) -> dict[str, object]:
        if False:
            for i in range(10):
                print('nop')
        'Stop daemon.'
        os.unlink(self.status_file)
        return {}

    def cmd_run(self, version: str, args: Sequence[str], export_types: bool, is_tty: bool, terminal_width: int) -> dict[str, object]:
        if False:
            print('Hello World!')
        'Check a list of files, triggering a restart if needed.'
        stderr = io.StringIO()
        stdout = io.StringIO()
        try:
            with redirect_stderr(stderr):
                with redirect_stdout(stdout):
                    (sources, options) = mypy.main.process_options(['-i'] + list(args), require_targets=True, server_options=True, fscache=self.fscache, program='mypy-daemon', header=argparse.SUPPRESS)
            if not options.compare_stable(self.options_snapshot):
                return {'restart': 'configuration changed'}
            if __version__ != version:
                return {'restart': 'mypy version changed'}
            if self.fine_grained_manager:
                manager = self.fine_grained_manager.manager
                start_plugins_snapshot = manager.plugins_snapshot
                (_, current_plugins_snapshot) = mypy.build.load_plugins(options, manager.errors, sys.stdout, extra_plugins=())
                if current_plugins_snapshot != start_plugins_snapshot:
                    return {'restart': 'plugins changed'}
        except InvalidSourceList as err:
            return {'out': '', 'err': str(err), 'status': 2}
        except SystemExit as e:
            return {'out': stdout.getvalue(), 'err': stderr.getvalue(), 'status': e.code}
        return self.check(sources, export_types, is_tty, terminal_width)

    def cmd_check(self, files: Sequence[str], export_types: bool, is_tty: bool, terminal_width: int) -> dict[str, object]:
        if False:
            i = 10
            return i + 15
        'Check a list of files.'
        try:
            sources = create_source_list(files, self.options, self.fscache)
        except InvalidSourceList as err:
            return {'out': '', 'err': str(err), 'status': 2}
        return self.check(sources, export_types, is_tty, terminal_width)

    def cmd_recheck(self, is_tty: bool, terminal_width: int, export_types: bool, remove: list[str] | None=None, update: list[str] | None=None) -> dict[str, object]:
        if False:
            return 10
        'Check the same list of files we checked most recently.\n\n        If remove/update is given, they modify the previous list;\n        if all are None, stat() is called for each file in the previous list.\n        '
        t0 = time.time()
        if not self.fine_grained_manager:
            return {'error': "Command 'recheck' is only valid after a 'check' command"}
        sources = self.previous_sources
        if remove:
            removals = set(remove)
            sources = [s for s in sources if s.path and s.path not in removals]
        if update:
            known = {s.path for s in sources if s.path}
            added = [p for p in update if p not in known]
            try:
                added_sources = create_source_list(added, self.options, self.fscache)
            except InvalidSourceList as err:
                return {'out': '', 'err': str(err), 'status': 2}
            sources = sources + added_sources
        t1 = time.time()
        manager = self.fine_grained_manager.manager
        manager.log(f'fine-grained increment: cmd_recheck: {t1 - t0:.3f}s')
        old_export_types = self.options.export_types
        self.options.export_types = self.options.export_types or export_types
        if not self.following_imports():
            messages = self.fine_grained_increment(sources, remove, update, explicit_export_types=export_types)
        else:
            assert remove is None and update is None
            messages = self.fine_grained_increment_follow_imports(sources, explicit_export_types=export_types)
        res = self.increment_output(messages, sources, is_tty, terminal_width)
        self.flush_caches()
        self.update_stats(res)
        self.options.export_types = old_export_types
        return res

    def check(self, sources: list[BuildSource], export_types: bool, is_tty: bool, terminal_width: int) -> dict[str, Any]:
        if False:
            print('Hello World!')
        'Check using fine-grained incremental mode.\n\n        If is_tty is True format the output nicely with colors and summary line\n        (unless disabled in self.options). Also pass the terminal_width to formatter.\n        '
        old_export_types = self.options.export_types
        self.options.export_types = self.options.export_types or export_types
        if not self.fine_grained_manager:
            res = self.initialize_fine_grained(sources, is_tty, terminal_width)
        else:
            if not self.following_imports():
                messages = self.fine_grained_increment(sources, explicit_export_types=export_types)
            else:
                messages = self.fine_grained_increment_follow_imports(sources, explicit_export_types=export_types)
            res = self.increment_output(messages, sources, is_tty, terminal_width)
        self.flush_caches()
        self.update_stats(res)
        self.options.export_types = old_export_types
        return res

    def flush_caches(self) -> None:
        if False:
            i = 10
            return i + 15
        self.fscache.flush()
        if self.fine_grained_manager:
            self.fine_grained_manager.flush_cache()

    def update_stats(self, res: dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.fine_grained_manager:
            manager = self.fine_grained_manager.manager
            manager.dump_stats()
            res['stats'] = manager.stats
            manager.stats = {}

    def following_imports(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Are we following imports?'
        return self.options.follow_imports == 'normal'

    def initialize_fine_grained(self, sources: list[BuildSource], is_tty: bool, terminal_width: int) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        self.fswatcher = FileSystemWatcher(self.fscache)
        t0 = time.time()
        self.update_sources(sources)
        t1 = time.time()
        try:
            result = mypy.build.build(sources=sources, options=self.options, fscache=self.fscache)
        except mypy.errors.CompileError as e:
            output = ''.join((s + '\n' for s in e.messages))
            if e.use_stdout:
                (out, err) = (output, '')
            else:
                (out, err) = ('', output)
            return {'out': out, 'err': err, 'status': 2}
        messages = result.errors
        self.fine_grained_manager = FineGrainedBuildManager(result)
        original_sources_len = len(sources)
        if self.following_imports():
            sources = find_all_sources_in_build(self.fine_grained_manager.graph, sources)
            self.update_sources(sources)
        self.previous_sources = sources
        if result.used_cache:
            t2 = time.time()
            for state in self.fine_grained_manager.graph.values():
                meta = state.meta
                if meta is None:
                    continue
                assert state.path is not None
                self.fswatcher.set_file_data(state.path, FileData(st_mtime=float(meta.mtime), st_size=meta.size, hash=meta.hash))
            (changed, removed) = self.find_changed(sources)
            changed += self.find_added_suppressed(self.fine_grained_manager.graph, set(), self.fine_grained_manager.manager.search_paths)
            for state in self.fine_grained_manager.graph.values():
                if not state.is_fresh():
                    assert state.path is not None
                    changed.append((state.id, state.path))
            t3 = time.time()
            messages = self.fine_grained_manager.update(changed, removed)
            if self.following_imports():
                messages = self.fine_grained_increment_follow_imports(sources)
            t4 = time.time()
            self.fine_grained_manager.manager.add_stats(update_sources_time=t1 - t0, build_time=t2 - t1, find_changes_time=t3 - t2, fg_update_time=t4 - t3, files_changed=len(removed) + len(changed))
        else:
            self.fswatcher.find_changed()
        if MEM_PROFILE:
            from mypy.memprofile import print_memory_profile
            print_memory_profile(run_gc=False)
        (__, n_notes, __) = count_stats(messages)
        status = 1 if messages and n_notes < len(messages) else 0
        messages = self.pretty_messages(messages, original_sources_len, is_tty, terminal_width)
        return {'out': ''.join((s + '\n' for s in messages)), 'err': '', 'status': status}

    def fine_grained_increment(self, sources: list[BuildSource], remove: list[str] | None=None, update: list[str] | None=None, explicit_export_types: bool=False) -> list[str]:
        if False:
            print('Hello World!')
        'Perform a fine-grained type checking increment.\n\n        If remove and update are None, determine changed paths by using\n        fswatcher. Otherwise, assume that only these files have changes.\n\n        Args:\n            sources: sources passed on the command line\n            remove: paths of files that have been removed\n            update: paths of files that have been changed or created\n            explicit_export_types: --export-type was passed in a check command\n              (as opposite to being set in dmypy start)\n        '
        assert self.fine_grained_manager is not None
        manager = self.fine_grained_manager.manager
        t0 = time.time()
        if remove is None and update is None:
            self.update_sources(sources)
            (changed, removed) = self.find_changed(sources)
        else:
            (changed, removed) = self.update_changed(sources, remove or [], update or [])
        if explicit_export_types:
            add_all_sources_to_changed(sources, changed)
        changed += self.find_added_suppressed(self.fine_grained_manager.graph, set(), manager.search_paths)
        manager.search_paths = compute_search_paths(sources, manager.options, manager.data_dir)
        t1 = time.time()
        manager.log(f'fine-grained increment: find_changed: {t1 - t0:.3f}s')
        messages = self.fine_grained_manager.update(changed, removed)
        t2 = time.time()
        manager.log(f'fine-grained increment: update: {t2 - t1:.3f}s')
        manager.add_stats(find_changes_time=t1 - t0, fg_update_time=t2 - t1, files_changed=len(removed) + len(changed))
        self.previous_sources = sources
        return messages

    def fine_grained_increment_follow_imports(self, sources: list[BuildSource], explicit_export_types: bool=False) -> list[str]:
        if False:
            print('Hello World!')
        'Like fine_grained_increment, but follow imports.'
        t0 = time.time()
        assert self.fine_grained_manager is not None
        fine_grained_manager = self.fine_grained_manager
        graph = fine_grained_manager.graph
        manager = fine_grained_manager.manager
        orig_modules = list(graph.keys())
        self.update_sources(sources)
        changed_paths = self.fswatcher.find_changed()
        manager.search_paths = compute_search_paths(sources, manager.options, manager.data_dir)
        t1 = time.time()
        manager.log(f'fine-grained increment: find_changed: {t1 - t0:.3f}s')
        seen = {source.module for source in sources}
        (changed, new_files) = self.find_reachable_changed_modules(sources, graph, seen, changed_paths)
        if explicit_export_types:
            add_all_sources_to_changed(sources, changed)
        sources.extend(new_files)
        messages = fine_grained_manager.update(changed, [], followed=True)
        worklist = changed.copy()
        while worklist:
            module = worklist.pop()
            if module[0] not in graph:
                continue
            sources2 = self.direct_imports(module, graph)
            sources2 = [source for source in sources2 if source.module not in seen]
            (changed, new_files) = self.find_reachable_changed_modules(sources2, graph, seen, changed_paths)
            self.update_sources(new_files)
            messages = fine_grained_manager.update(changed, [], followed=True)
            worklist.extend(changed)
        t2 = time.time()

        def refresh_file(module: str, path: str) -> list[str]:
            if False:
                i = 10
                return i + 15
            return fine_grained_manager.update([(module, path)], [], followed=True)
        for (module_id, state) in list(graph.items()):
            new_messages = refresh_suppressed_submodules(module_id, state.path, fine_grained_manager.deps, graph, self.fscache, refresh_file)
            if new_messages is not None:
                messages = new_messages
        t3 = time.time()
        while True:
            new_unsuppressed = self.find_added_suppressed(graph, seen, manager.search_paths)
            if not new_unsuppressed:
                break
            new_files = [BuildSource(mod[1], mod[0], followed=True) for mod in new_unsuppressed]
            sources.extend(new_files)
            self.update_sources(new_files)
            messages = fine_grained_manager.update(new_unsuppressed, [], followed=True)
            for (module_id, path) in new_unsuppressed:
                new_messages = refresh_suppressed_submodules(module_id, path, fine_grained_manager.deps, graph, self.fscache, refresh_file)
                if new_messages is not None:
                    messages = new_messages
        t4 = time.time()
        to_delete = []
        for module_id in orig_modules:
            if module_id not in graph:
                continue
            if module_id not in seen:
                module_path = graph[module_id].path
                assert module_path is not None
                to_delete.append((module_id, module_path))
        if to_delete:
            messages = fine_grained_manager.update([], to_delete)
        fix_module_deps(graph)
        self.previous_sources = find_all_sources_in_build(graph)
        self.update_sources(self.previous_sources)
        self.fswatcher.find_changed()
        t5 = time.time()
        manager.log(f'fine-grained increment: update: {t5 - t1:.3f}s')
        manager.add_stats(find_changes_time=t1 - t0, fg_update_time=t2 - t1, refresh_suppressed_time=t3 - t2, find_added_supressed_time=t4 - t3, cleanup_time=t5 - t4)
        return messages

    def find_reachable_changed_modules(self, roots: list[BuildSource], graph: mypy.build.Graph, seen: set[str], changed_paths: AbstractSet[str]) -> tuple[list[tuple[str, str]], list[BuildSource]]:
        if False:
            return 10
        "Follow imports within graph from given sources until hitting changed modules.\n\n        If we find a changed module, we can't continue following imports as the imports\n        may have changed.\n\n        Args:\n            roots: modules where to start search from\n            graph: module graph to use for the search\n            seen: modules we've seen before that won't be visited (mutated here!!)\n            changed_paths: which paths have changed (stop search here and return any found)\n\n        Return (encountered reachable changed modules,\n                unchanged files not in sources_set traversed).\n        "
        changed = []
        new_files = []
        worklist = roots.copy()
        seen.update((source.module for source in worklist))
        while worklist:
            nxt = worklist.pop()
            if nxt.module not in seen:
                seen.add(nxt.module)
                new_files.append(nxt)
            if nxt.path in changed_paths:
                assert nxt.path is not None
                changed.append((nxt.module, nxt.path))
            elif nxt.module in graph:
                state = graph[nxt.module]
                for dep in state.dependencies:
                    if dep not in seen:
                        seen.add(dep)
                        worklist.append(BuildSource(graph[dep].path, graph[dep].id, followed=True))
        return (changed, new_files)

    def direct_imports(self, module: tuple[str, str], graph: mypy.build.Graph) -> list[BuildSource]:
        if False:
            i = 10
            return i + 15
        'Return the direct imports of module not included in seen.'
        state = graph[module[0]]
        return [BuildSource(graph[dep].path, dep, followed=True) for dep in state.dependencies]

    def find_added_suppressed(self, graph: mypy.build.Graph, seen: set[str], search_paths: SearchPaths) -> list[tuple[str, str]]:
        if False:
            i = 10
            return i + 15
        "Find suppressed modules that have been added (and not included in seen).\n\n        Args:\n            seen: reachable modules we've seen before (mutated here!!)\n\n        Return suppressed, added modules.\n        "
        all_suppressed = set()
        for state in graph.values():
            all_suppressed |= state.suppressed_set
        all_suppressed = {module for module in all_suppressed if module not in graph and (not ignore_suppressed_imports(module))}
        packages = {module.split('.', 1)[0] for module in all_suppressed}
        packages = filter_out_missing_top_level_packages(packages, search_paths, self.fscache)
        finder = FindModuleCache(search_paths, self.fscache, self.options)
        found = []
        for module in all_suppressed:
            top_level_pkg = module.split('.', 1)[0]
            if top_level_pkg not in packages:
                continue
            result = finder.find_module(module, fast_path=True)
            if isinstance(result, str) and module not in seen:
                if not self.following_imports() and (not result.endswith('.pyi')):
                    continue
                found.append((module, result))
                seen.add(module)
        return found

    def increment_output(self, messages: list[str], sources: list[BuildSource], is_tty: bool, terminal_width: int) -> dict[str, Any]:
        if False:
            print('Hello World!')
        status = 1 if messages else 0
        messages = self.pretty_messages(messages, len(sources), is_tty, terminal_width)
        return {'out': ''.join((s + '\n' for s in messages)), 'err': '', 'status': status}

    def pretty_messages(self, messages: list[str], n_sources: int, is_tty: bool=False, terminal_width: int | None=None) -> list[str]:
        if False:
            i = 10
            return i + 15
        use_color = self.options.color_output and is_tty
        fit_width = self.options.pretty and is_tty
        if fit_width:
            messages = self.formatter.fit_in_terminal(messages, fixed_terminal_width=terminal_width)
        if self.options.error_summary:
            summary: str | None = None
            (n_errors, n_notes, n_files) = count_stats(messages)
            if n_errors:
                summary = self.formatter.format_error(n_errors, n_files, n_sources, use_color=use_color)
            elif not messages or n_notes == len(messages):
                summary = self.formatter.format_success(n_sources, use_color)
            if summary:
                messages = messages + [summary]
        if use_color:
            messages = [self.formatter.colorize(m) for m in messages]
        return messages

    def update_sources(self, sources: list[BuildSource]) -> None:
        if False:
            i = 10
            return i + 15
        paths = [source.path for source in sources if source.path is not None]
        if self.following_imports():
            paths = [path for path in paths if self.fscache.isfile(path)]
        self.fswatcher.add_watched_paths(paths)

    def update_changed(self, sources: list[BuildSource], remove: list[str], update: list[str]) -> ChangesAndRemovals:
        if False:
            for i in range(10):
                print('nop')
        changed_paths = self.fswatcher.update_changed(remove, update)
        return self._find_changed(sources, changed_paths)

    def find_changed(self, sources: list[BuildSource]) -> ChangesAndRemovals:
        if False:
            while True:
                i = 10
        changed_paths = self.fswatcher.find_changed()
        return self._find_changed(sources, changed_paths)

    def _find_changed(self, sources: list[BuildSource], changed_paths: AbstractSet[str]) -> ChangesAndRemovals:
        if False:
            i = 10
            return i + 15
        changed = [(source.module, source.path) for source in sources if source.path and source.path in changed_paths]
        modules = {source.module for source in sources}
        omitted = [source for source in self.previous_sources if source.module not in modules]
        removed = []
        for source in omitted:
            path = source.path
            assert path
            removed.append((source.module, path))
        previous_modules = {source.module for source in self.previous_sources}
        changed_set = set(changed)
        changed.extend([(source.module, source.path) for source in sources if source.path and source.module not in previous_modules and ((source.module, source.path) not in changed_set)])
        last = {s.path: s.module for s in self.previous_sources}
        for s in sources:
            assert s.path
            if s.path in last and last[s.path] != s.module:
                removed.append((last[s.path], s.path))
                changed.append((s.module, s.path))
        return (changed, removed)

    def cmd_inspect(self, show: str, location: str, verbosity: int=0, limit: int=0, include_span: bool=False, include_kind: bool=False, include_object_attrs: bool=False, union_attrs: bool=False, force_reload: bool=False) -> dict[str, object]:
        if False:
            while True:
                i = 10
        'Locate and inspect expression(s).'
        if not self.fine_grained_manager:
            return {'error': 'Command "inspect" is only valid after a "check" command (that produces no parse errors)'}
        engine = InspectionEngine(self.fine_grained_manager, verbosity=verbosity, limit=limit, include_span=include_span, include_kind=include_kind, include_object_attrs=include_object_attrs, union_attrs=union_attrs, force_reload=force_reload)
        old_inspections = self.options.inspections
        self.options.inspections = True
        try:
            if show == 'type':
                result = engine.get_type(location)
            elif show == 'attrs':
                result = engine.get_attrs(location)
            elif show == 'definition':
                result = engine.get_definition(location)
            else:
                assert False, 'Unknown inspection kind'
        finally:
            self.options.inspections = old_inspections
        if 'out' in result:
            assert isinstance(result['out'], str)
            result['out'] += '\n'
        return result

    def cmd_suggest(self, function: str, callsites: bool, **kwargs: Any) -> dict[str, object]:
        if False:
            while True:
                i = 10
        'Suggest a signature for a function.'
        if not self.fine_grained_manager:
            return {'error': "Command 'suggest' is only valid after a 'check' command (that produces no parse errors)"}
        engine = SuggestionEngine(self.fine_grained_manager, **kwargs)
        try:
            if callsites:
                out = engine.suggest_callsites(function)
            else:
                out = engine.suggest(function)
        except SuggestionFailure as err:
            return {'error': str(err)}
        else:
            if not out:
                out = 'No suggestions\n'
            elif not out.endswith('\n'):
                out += '\n'
            return {'out': out, 'err': '', 'status': 0}
        finally:
            self.flush_caches()

    def cmd_hang(self) -> dict[str, object]:
        if False:
            return 10
        'Hang for 100 seconds, as a debug hack.'
        time.sleep(100)
        return {}
MiB: Final = 2 ** 20

def get_meminfo() -> dict[str, Any]:
    if False:
        while True:
            i = 10
    res: dict[str, Any] = {}
    try:
        import psutil
    except ImportError:
        res['memory_psutil_missing'] = 'psutil not found, run pip install mypy[dmypy] to install the needed components for dmypy'
    else:
        process = psutil.Process()
        meminfo = process.memory_info()
        res['memory_rss_mib'] = meminfo.rss / MiB
        res['memory_vms_mib'] = meminfo.vms / MiB
        if sys.platform == 'win32':
            res['memory_maxrss_mib'] = meminfo.peak_wset / MiB
        else:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == 'darwin':
                factor = 1
            else:
                factor = 1024
            res['memory_maxrss_mib'] = rusage.ru_maxrss * factor / MiB
    return res

def find_all_sources_in_build(graph: mypy.build.Graph, extra: Sequence[BuildSource]=()) -> list[BuildSource]:
    if False:
        print('Hello World!')
    result = list(extra)
    seen = {source.module for source in result}
    for (module, state) in graph.items():
        if module not in seen:
            result.append(BuildSource(state.path, module))
    return result

def add_all_sources_to_changed(sources: list[BuildSource], changed: list[tuple[str, str]]) -> None:
    if False:
        while True:
            i = 10
    'Add all (explicit) sources to the list changed files in place.\n\n    Use this when re-processing of unchanged files is needed (e.g. for\n    the purpose of exporting types for inspections).\n    '
    changed_set = set(changed)
    changed.extend([(bs.module, bs.path) for bs in sources if bs.path and (bs.module, bs.path) not in changed_set])

def fix_module_deps(graph: mypy.build.Graph) -> None:
    if False:
        return 10
    'After an incremental update, update module dependencies to reflect the new state.\n\n    This can make some suppressed dependencies non-suppressed, and vice versa (if modules\n    have been added to or removed from the build).\n    '
    for (module, state) in graph.items():
        new_suppressed = []
        new_dependencies = []
        for dep in state.dependencies + state.suppressed:
            if dep in graph:
                new_dependencies.append(dep)
            else:
                new_suppressed.append(dep)
        state.dependencies = new_dependencies
        state.dependencies_set = set(new_dependencies)
        state.suppressed = new_suppressed
        state.suppressed_set = set(new_suppressed)

def filter_out_missing_top_level_packages(packages: set[str], search_paths: SearchPaths, fscache: FileSystemCache) -> set[str]:
    if False:
        i = 10
        return i + 15
    "Quickly filter out obviously missing top-level packages.\n\n    Return packages with entries that can't be found removed.\n\n    This is approximate: some packages that aren't actually valid may be\n    included. However, all potentially valid packages must be returned.\n    "
    found = set()
    paths = search_paths.python_path + search_paths.mypy_path + search_paths.package_path + search_paths.typeshed_path
    for p in paths:
        try:
            entries = fscache.listdir(p)
        except Exception:
            entries = []
        for entry in entries:
            if entry.endswith('.py'):
                entry = entry[:-3]
            elif entry.endswith('.pyi'):
                entry = entry[:-4]
            elif entry.endswith('-stubs'):
                entry = entry[:-6]
            if entry in packages:
                found.add(entry)
    return found