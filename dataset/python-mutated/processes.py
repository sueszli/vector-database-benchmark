import os
from glances.globals import BSD, LINUX, MACOS, WINDOWS, iterkeys
from glances.timer import Timer, getTimeSinceLastUpdate
from glances.filter import GlancesFilter
from glances.programs import processes_to_programs
from glances.logger import logger
import psutil
sort_processes_key_list = ['cpu_percent', 'memory_percent', 'username', 'cpu_times', 'io_counters', 'name']
sort_for_human = {'io_counters': 'disk IO', 'cpu_percent': 'CPU consumption', 'memory_percent': 'memory consumption', 'cpu_times': 'process time', 'username': 'user name', 'name': 'processs name', None: 'None'}

class GlancesProcesses(object):
    """Get processed stats using the psutil library."""

    def __init__(self, cache_timeout=60):
        if False:
            i = 10
            return i + 15
        'Init the class to collect stats about processes.'
        self.args = None
        self.username_cache = {}
        self.cmdline_cache = {}
        self.cache_timeout = cache_timeout
        self.cache_timer = Timer(0)
        self.io_old = {}
        self.auto_sort = None
        self._sort_key = None
        self.set_sort_key('auto', auto=True)
        self.processlist = []
        self.reset_processcount()
        self.processlist_cache = {}
        self.disable_tag = False
        self.disable_extended_tag = False
        self.extended_process = None
        try:
            p = psutil.Process()
            p.io_counters()
        except Exception as e:
            logger.warning('PsUtil can not grab processes io_counters ({})'.format(e))
            self.disable_io_counters = True
        else:
            logger.debug('PsUtil can grab processes io_counters')
            self.disable_io_counters = False
        try:
            p = psutil.Process()
            p.gids()
        except Exception as e:
            logger.warning('PsUtil can not grab processes gids ({})'.format(e))
            self.disable_gids = True
        else:
            logger.debug('PsUtil can grab processes gids')
            self.disable_gids = False
        self._max_processes = None
        self._filter = GlancesFilter()
        self.no_kernel_threads = False
        self._max_values_list = ('cpu_percent', 'memory_percent')
        self._max_values = {}
        self.reset_max_values()

    def set_args(self, args):
        if False:
            return 10
        'Set args.'
        self.args = args

    def reset_processcount(self):
        if False:
            return 10
        'Reset the global process count'
        self.processcount = {'total': 0, 'running': 0, 'sleeping': 0, 'thread': 0, 'pid_max': None}

    def update_processcount(self, plist):
        if False:
            return 10
        'Update the global process count from the current processes list'
        self.processcount['pid_max'] = self.pid_max
        for k in iterkeys(self.processcount):
            self.processcount[k] = len(list(filter(lambda v: v['status'] is k, plist)))
        self.processcount['thread'] = sum((i['num_threads'] for i in plist if i['num_threads'] is not None))
        self.processcount['total'] = len(plist)

    def enable(self):
        if False:
            i = 10
            return i + 15
        'Enable process stats.'
        self.disable_tag = False
        self.update()

    def disable(self):
        if False:
            return 10
        'Disable process stats.'
        self.disable_tag = True

    def enable_extended(self):
        if False:
            i = 10
            return i + 15
        'Enable extended process stats.'
        self.disable_extended_tag = False
        self.update()

    def disable_extended(self):
        if False:
            return 10
        'Disable extended process stats.'
        self.disable_extended_tag = True

    @property
    def pid_max(self):
        if False:
            print('Hello World!')
        '\n        Get the maximum PID value.\n\n        On Linux, the value is read from the `/proc/sys/kernel/pid_max` file.\n\n        From `man 5 proc`:\n        The default value for this file, 32768, results in the same range of\n        PIDs as on earlier kernels. On 32-bit platforms, 32768 is the maximum\n        value for pid_max. On 64-bit systems, pid_max can be set to any value\n        up to 2^22 (PID_MAX_LIMIT, approximately 4 million).\n\n        If the file is unreadable or not available for whatever reason,\n        returns None.\n\n        Some other OSes:\n        - On FreeBSD and macOS the maximum is 99999.\n        - On OpenBSD >= 6.0 the maximum is 99999 (was 32766).\n        - On NetBSD the maximum is 30000.\n\n        :returns: int or None\n        '
        if LINUX:
            try:
                with open('/proc/sys/kernel/pid_max', 'rb') as f:
                    return int(f.read())
            except (OSError, IOError):
                return None
        else:
            return None

    @property
    def processes_count(self):
        if False:
            print('Hello World!')
        'Get the current number of processes showed in the UI.'
        return min(self._max_processes - 2, glances_processes.processcount['total'] - 1)

    @property
    def max_processes(self):
        if False:
            while True:
                i = 10
        'Get the maximum number of processes showed in the UI.'
        return self._max_processes

    @max_processes.setter
    def max_processes(self, value):
        if False:
            print('Hello World!')
        'Set the maximum number of processes showed in the UI.'
        self._max_processes = value

    @property
    def process_filter_input(self):
        if False:
            while True:
                i = 10
        'Get the process filter (given by the user).'
        return self._filter.filter_input

    @property
    def process_filter(self):
        if False:
            i = 10
            return i + 15
        'Get the process filter (current apply filter).'
        return self._filter.filter

    @process_filter.setter
    def process_filter(self, value):
        if False:
            while True:
                i = 10
        'Set the process filter.'
        self._filter.filter = value

    @property
    def process_filter_key(self):
        if False:
            while True:
                i = 10
        'Get the process filter key.'
        return self._filter.filter_key

    @property
    def process_filter_re(self):
        if False:
            return 10
        'Get the process regular expression compiled.'
        return self._filter.filter_re

    def disable_kernel_threads(self):
        if False:
            return 10
        'Ignore kernel threads in process list.'
        self.no_kernel_threads = True

    @property
    def sort_reverse(self):
        if False:
            return 10
        "Return True to sort processes in reverse 'key' order, False instead."
        if self.sort_key == 'name' or self.sort_key == 'username':
            return False
        return True

    def max_values(self):
        if False:
            while True:
                i = 10
        'Return the max values dict.'
        return self._max_values

    def get_max_values(self, key):
        if False:
            while True:
                i = 10
        'Get the maximum values of the given stat (key).'
        return self._max_values[key]

    def set_max_values(self, key, value):
        if False:
            while True:
                i = 10
        'Set the maximum value for a specific stat (key).'
        self._max_values[key] = value

    def reset_max_values(self):
        if False:
            while True:
                i = 10
        'Reset the maximum values dict.'
        self._max_values = {}
        for k in self._max_values_list:
            self._max_values[k] = 0.0

    def get_extended_stats(self, proc):
        if False:
            return 10
        'Get the extended stats for the given PID.'
        extended_stats = ['cpu_affinity', 'ionice', 'num_ctx_switches']
        if LINUX:
            extended_stats += ['num_fds']
        if WINDOWS:
            extended_stats += ['num_handles']
        ret = {}
        try:
            selected_process = psutil.Process(proc['pid'])
            ret = selected_process.as_dict(attrs=extended_stats, ad_value=None)
            if LINUX:
                try:
                    ret['memory_swap'] = sum([v.swap for v in selected_process.memory_maps()])
                except (psutil.NoSuchProcess, KeyError):
                    pass
                except (psutil.AccessDenied, NotImplementedError):
                    ret['memory_swap'] = None
            try:
                ret['tcp'] = len(selected_process.connections(kind='tcp'))
                ret['udp'] = len(selected_process.connections(kind='udp'))
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                ret['tcp'] = None
                ret['udp'] = None
        except (psutil.NoSuchProcess, ValueError, AttributeError) as e:
            logger.error('Can not grab extended stats ({})'.format(e))
            self.extended_process = None
            ret['extended_stats'] = False
        else:
            logger.debug('Grab extended stats for process {}'.format(proc['pid']))
            for stat_prefix in ['cpu', 'memory']:
                if stat_prefix + '_min' not in self.extended_process:
                    ret[stat_prefix + '_min'] = proc[stat_prefix + '_percent']
                else:
                    ret[stat_prefix + '_min'] = proc[stat_prefix + '_percent'] if proc[stat_prefix + '_min'] > proc[stat_prefix + '_percent'] else proc[stat_prefix + '_min']
                if stat_prefix + '_max' not in self.extended_process:
                    ret[stat_prefix + '_max'] = proc[stat_prefix + '_percent']
                else:
                    ret[stat_prefix + '_max'] = proc[stat_prefix + '_percent'] if proc[stat_prefix + '_max'] < proc[stat_prefix + '_percent'] else proc[stat_prefix + '_max']
                if stat_prefix + '_mean_sum' not in self.extended_process:
                    ret[stat_prefix + '_mean_sum'] = proc[stat_prefix + '_percent']
                else:
                    ret[stat_prefix + '_mean_sum'] = proc[stat_prefix + '_mean_sum'] + proc[stat_prefix + '_percent']
                if stat_prefix + '_mean_counter' not in self.extended_process:
                    ret[stat_prefix + '_mean_counter'] = 1
                else:
                    ret[stat_prefix + '_mean_counter'] = proc[stat_prefix + '_mean_counter'] + 1
                ret[stat_prefix + '_mean'] = ret[stat_prefix + '_mean_sum'] / ret[stat_prefix + '_mean_counter']
            ret['extended_stats'] = True
        return ret

    def is_selected_extended_process(self, position):
        if False:
            print('Hello World!')
        'Return True if the process is the selected one for extended stats.'
        return hasattr(self.args, 'programs') and (not self.args.programs) and hasattr(self.args, 'enable_process_extended') and self.args.enable_process_extended and (not self.disable_extended_tag) and hasattr(self.args, 'cursor_position') and (position == self.args.cursor_position) and (not self.args.disable_cursor)

    def update(self):
        if False:
            while True:
                i = 10
        'Update the processes stats.'
        self.processlist = []
        self.reset_processcount()
        if self.disable_tag:
            return
        time_since_update = getTimeSinceLastUpdate('process_disk')
        sorted_attrs = ['cpu_percent', 'cpu_times', 'memory_percent', 'name', 'status', 'num_threads']
        displayed_attr = ['memory_info', 'nice', 'pid']
        cached_attrs = ['cmdline', 'username']
        if not self.disable_io_counters:
            sorted_attrs.append('io_counters')
        if not self.disable_gids:
            displayed_attr.append('gids')
        sorted_attrs.extend(displayed_attr)
        if self.cache_timer.finished():
            sorted_attrs += cached_attrs
            self.cache_timer.set(self.cache_timeout)
            self.cache_timer.reset()
            is_cached = False
        else:
            is_cached = True
        self.processlist = list(filter(lambda p: not (BSD and p.info['name'] == 'idle') and (not (WINDOWS and p.info['name'] == 'System Idle Process')) and (not (MACOS and p.info['name'] == 'kernel_task')) and (not (self.no_kernel_threads and LINUX and (p.info['gids'].real == 0))), psutil.process_iter(attrs=sorted_attrs, ad_value=None)))
        self.processlist = [p.info for p in self.processlist]
        self.processlist = sort_stats(self.processlist, sorted_by=self.sort_key, reverse=True)
        self.update_processcount(self.processlist)
        for (position, proc) in enumerate(self.processlist):
            if self.is_selected_extended_process(position):
                self.extended_process = proc
            if self.extended_process is not None and proc['pid'] == self.extended_process['pid']:
                proc.update(self.get_extended_stats(self.extended_process))
                self.extended_process = proc
            proc['key'] = 'pid'
            proc['time_since_update'] = time_since_update
            proc['status'] = str(proc['status'])[:1].upper()
            if 'io_counters' in proc and proc['io_counters'] is not None:
                io_new = [proc['io_counters'].read_bytes, proc['io_counters'].write_bytes]
                try:
                    proc['io_counters'] = io_new + self.io_old[proc['pid']]
                    io_tag = 1
                except KeyError:
                    proc['io_counters'] = io_new + [0, 0]
                    io_tag = 0
                self.io_old[proc['pid']] = io_new
            else:
                proc['io_counters'] = [0, 0] + [0, 0]
                io_tag = 0
            proc['io_counters'] += [io_tag]
            if is_cached:
                if proc['pid'] not in self.processlist_cache:
                    try:
                        self.processlist_cache[proc['pid']] = psutil.Process(pid=proc['pid']).as_dict(attrs=cached_attrs, ad_value=None)
                    except psutil.NoSuchProcess:
                        pass
                try:
                    proc.update(self.processlist_cache[proc['pid']])
                except KeyError:
                    pass
            else:
                self.processlist_cache[proc['pid']] = {cached: proc[cached] for cached in cached_attrs}
        self.processlist = list(filter(lambda p: not self._filter.is_filtered(p), self.processlist))
        for k in self._max_values_list:
            values_list = [i[k] for i in self.processlist if i[k] is not None]
            if values_list:
                self.set_max_values(k, max(values_list))

    def get_count(self):
        if False:
            i = 10
            return i + 15
        'Get the number of processes.'
        return self.processcount

    def getlist(self, sorted_by=None, as_programs=False):
        if False:
            print('Hello World!')
        'Get the processlist.\n        By default, return the list of threads.\n        If as_programs is True, return the list of programs.'
        if as_programs:
            return processes_to_programs(self.processlist)
        else:
            return self.processlist

    @property
    def sort_key(self):
        if False:
            while True:
                i = 10
        'Get the current sort key.'
        return self._sort_key

    def set_sort_key(self, key, auto=True):
        if False:
            for i in range(10):
                print('nop')
        'Set the current sort key.'
        if key == 'auto':
            self.auto_sort = True
            self._sort_key = 'cpu_percent'
        else:
            self.auto_sort = auto
            self._sort_key = key

    def nice_decrease(self, pid):
        if False:
            print('Hello World!')
        'Decrease nice level\n        On UNIX this is a number which usually goes from -20 to 20.\n        The higher the nice value, the lower the priority of the process.'
        p = psutil.Process(pid)
        try:
            p.nice(p.nice() - 1)
            logger.info('Set nice level of process {} to {} (higher the priority)'.format(pid, p.nice()))
        except psutil.AccessDenied:
            logger.warning('Can not decrease (higher the priority) the nice level of process {} (access denied)'.format(pid))

    def nice_increase(self, pid):
        if False:
            while True:
                i = 10
        'Increase nice level\n        On UNIX this is a number which usually goes from -20 to 20.\n        The higher the nice value, the lower the priority of the process.'
        p = psutil.Process(pid)
        try:
            p.nice(p.nice() + 1)
            logger.info('Set nice level of process {} to {} (lower the priority)'.format(pid, p.nice()))
        except psutil.AccessDenied:
            logger.warning('Can not increase (lower the priority) the nice level of process {} (access denied)'.format(pid))

    def kill(self, pid, timeout=3):
        if False:
            i = 10
            return i + 15
        'Kill process with pid'
        assert pid != os.getpid(), 'Glances can kill itself...'
        p = psutil.Process(pid)
        logger.debug('Send kill signal to process: {}'.format(p))
        p.kill()
        return p.wait(timeout)

def weighted(value):
    if False:
        return 10
    'Manage None value in dict value.'
    return -float('inf') if value is None else value

def _sort_io_counters(process, sorted_by='io_counters', sorted_by_secondary='memory_percent'):
    if False:
        return 10
    'Specific case for io_counters\n\n    :return: Sum of io_r + io_w\n    '
    return process[sorted_by][0] - process[sorted_by][2] + process[sorted_by][1] - process[sorted_by][3]

def _sort_cpu_times(process, sorted_by='cpu_times', sorted_by_secondary='memory_percent'):
    if False:
        i = 10
        return i + 15
    'Specific case for cpu_times\n\n    Patch for "Sorting by process time works not as expected #1321"\n    By default PsUtil only takes user time into account\n    see (https://github.com/giampaolo/psutil/issues/1339)\n    The following implementation takes user and system time into account\n    '
    return process[sorted_by][0] + process[sorted_by][1]

def _sort_lambda(sorted_by='cpu_percent', sorted_by_secondary='memory_percent'):
    if False:
        i = 10
        return i + 15
    'Return a sort lambda function for the sorted_by key'
    ret = None
    if sorted_by == 'io_counters':
        ret = _sort_io_counters
    elif sorted_by == 'cpu_times':
        ret = _sort_cpu_times
    return ret

def sort_stats(stats, sorted_by='cpu_percent', sorted_by_secondary='memory_percent', reverse=True):
    if False:
        print('Hello World!')
    'Return the stats (dict) sorted by (sorted_by).\n\n    Reverse the sort if reverse is True.\n    '
    if sorted_by is None and sorted_by_secondary is None:
        return stats
    sort_lambda = _sort_lambda(sorted_by=sorted_by, sorted_by_secondary=sorted_by_secondary)
    if sort_lambda is not None:
        try:
            stats.sort(key=sort_lambda, reverse=reverse)
        except Exception:
            stats.sort(key=lambda process: (weighted(process['cpu_percent']), weighted(process[sorted_by_secondary])), reverse=reverse)
    else:
        try:
            stats.sort(key=lambda process: (weighted(process[sorted_by]), weighted(process[sorted_by_secondary])), reverse=reverse)
        except (KeyError, TypeError):
            stats.sort(key=lambda process: process['name'] if process['name'] is not None else '~', reverse=False)
    return stats
glances_processes = GlancesProcesses()