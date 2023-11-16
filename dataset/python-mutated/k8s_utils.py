import logging
from ray._private.utils import get_num_cpus
logger = logging.getLogger(__name__)
CPU_USAGE_PATH = '/sys/fs/cgroup/cpuacct/cpuacct.usage'
CPU_USAGE_PATH_V2 = '/sys/fs/cgroup/cpu.stat'
PROC_STAT_PATH = '/proc/stat'
container_num_cpus = None
host_num_cpus = None
last_cpu_usage = None
last_system_usage = None

def cpu_percent():
    if False:
        return 10
    "Estimate CPU usage percent for Ray pod managed by Kubernetes\n    Operator.\n\n    Computed by the following steps\n       (1) Replicate the logic used by 'docker stats' cli command.\n           See https://github.com/docker/cli/blob/c0a6b1c7b30203fbc28cd619acb901a95a80e30e/cli/command/container/stats_helpers.go#L166.\n       (2) Divide by the number of CPUs available to the container, so that\n           e.g. full capacity use of 2 CPUs will read as 100%,\n           rather than 200%.\n\n    Step (1) above works by\n        dividing delta in cpu usage by\n        delta in total host cpu usage, averaged over host's cpus.\n\n    Since deltas are not initially available, return 0.0 on first call.\n    "
    global last_system_usage
    global last_cpu_usage
    try:
        cpu_usage = _cpu_usage()
        system_usage = _system_usage()
        if last_system_usage is None:
            cpu_percent = 0.0
        else:
            cpu_delta = cpu_usage - last_cpu_usage
            system_delta = (system_usage - last_system_usage) / _host_num_cpus()
            quotient = cpu_delta / system_delta
            cpu_percent = round(quotient * 100 / get_num_cpus(), 1)
        last_system_usage = system_usage
        last_cpu_usage = cpu_usage
        return min(cpu_percent, 100.0)
    except Exception:
        logger.exception('Error computing CPU usage of Ray Kubernetes pod.')
        return 0.0

def _cpu_usage():
    if False:
        print('Hello World!')
    'Compute total cpu usage of the container in nanoseconds\n    by reading from cpuacct in cgroups v1 or cpu.stat in cgroups v2.'
    try:
        return int(open(CPU_USAGE_PATH).read())
    except FileNotFoundError:
        cpu_stat_text = open(CPU_USAGE_PATH_V2).read()
        cpu_stat_first_line = cpu_stat_text.split('\n')[0]
        cpu_usec = int(cpu_stat_first_line.split()[1])
        return cpu_usec * 1000

def _system_usage():
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes total CPU usage of the host in nanoseconds.\n\n    Logic taken from here:\n    https://github.com/moby/moby/blob/b42ac8d370a8ef8ec720dff0ca9dfb3530ac0a6a/daemon/stats/collector_unix.go#L31\n\n    See also the /proc/stat entry here:\n    https://man7.org/linux/man-pages/man5/proc.5.html\n    '
    cpu_summary_str = open(PROC_STAT_PATH).read().split('\n')[0]
    parts = cpu_summary_str.split()
    assert parts[0] == 'cpu'
    usage_data = parts[1:8]
    total_clock_ticks = sum((int(entry) for entry in usage_data))
    usage_ns = total_clock_ticks * 10 ** 7
    return usage_ns

def _host_num_cpus():
    if False:
        while True:
            i = 10
    'Number of physical CPUs, obtained by parsing /proc/stat.'
    global host_num_cpus
    if host_num_cpus is None:
        proc_stat_lines = open(PROC_STAT_PATH).read().split('\n')
        split_proc_stat_lines = [line.split() for line in proc_stat_lines]
        cpu_lines = [split_line for split_line in split_proc_stat_lines if len(split_line) > 0 and 'cpu' in split_line[0]]
        host_num_cpus = len(cpu_lines) - 1
    return host_num_cpus