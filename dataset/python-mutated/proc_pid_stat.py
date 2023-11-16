"""jc - JSON Convert `/proc/<pid>/stat` file parser

Usage (cli):

    $ cat /proc/1/stat | jc --proc

or

    $ jc /proc/1/stat

or

    $ cat /proc/1/stat | jc --proc-pid-stat

Usage (module):

    import jc
    result = jc.parse('proc', proc_pid_stat_file)

or

    import jc
    result = jc.parse('proc_pid_stat', proc_pid_stat_file)

Schema:

    {
      "pid":                            integer,
      "comm":                           string,
      "state":                          string,
      "state_pretty":                   string,
      "ppid":                           integer,
      "pgrp":                           integer,
      "session":                        integer,
      "tty_nr":                         integer,
      "tpg_id":                         integer,
      "flags":                          integer,
      "minflt":                         integer,
      "cminflt":                        integer,
      "majflt":                         integer,
      "cmajflt":                        integer,
      "utime":                          integer,
      "stime":                          integer,
      "cutime":                         integer,
      "cstime":                         integer,
      "priority":                       integer,
      "nice":                           integer,
      "num_threads":                    integer,
      "itrealvalue":                    integer,
      "starttime":                      integer,
      "vsize":                          integer,
      "rss":                            integer,
      "rsslim":                         integer,
      "startcode":                      integer,
      "endcode":                        integer,
      "startstack":                     integer,
      "kstkeep":                        integer,
      "kstkeip":                        integer,
      "signal":                         integer,
      "blocked":                        integer,
      "sigignore":                      integer,
      "sigcatch":                       integer,
      "wchan":                          integer,
      "nswap":                          integer,
      "cnswap":                         integer,
      "exit_signal":                    integer,
      "processor":                      integer,
      "rt_priority":                    integer,
      "policy":                         integer,
      "delayacct_blkio_ticks":          integer,
      "guest_time":                     integer,
      "cguest_time":                    integer,
      "start_data":                     integer,
      "end_data":                       integer,
      "start_brk":                      integer,
      "arg_start":                      integer,
      "arg_end":                        integer,
      "env_start":                      integer,
      "env_end":                        integer,
      "exit_code":                      integer,
    }

Examples:

    $ cat /proc/1/stat | jc --proc -p
    {
      "pid": 1,
      "comm": "systemd",
      "state": "S",
      "ppid": 0,
      "pgrp": 1,
      "session": 1,
      "tty_nr": 0,
      "tpg_id": -1,
      "flags": 4194560,
      "minflt": 23478,
      "cminflt": 350218,
      "majflt": 99,
      "cmajflt": 472,
      "utime": 107,
      "stime": 461,
      "cutime": 2672,
      "cstime": 4402,
      "priority": 20,
      "nice": 0,
      "num_threads": 1,
      "itrealvalue": 0,
      "starttime": 128,
      "vsize": 174063616,
      "rss": 3313,
      "rsslim": 18446744073709551615,
      "startcode": 94188219072512,
      "endcode": 94188219899461,
      "startstack": 140725059845296,
      "kstkeep": 0,
      "kstkeip": 0,
      "signal": 0,
      "blocked": 671173123,
      "sigignore": 4096,
      "sigcatch": 1260,
      "wchan": 1,
      "nswap": 0,
      "cnswap": 0,
      "exit_signal": 17,
      "processor": 0,
      "rt_priority": 0,
      "policy": 0,
      "delayacct_blkio_ticks": 18,
      "guest_time": 0,
      "cguest_time": 0,
      "start_data": 94188220274448,
      "end_data": 94188220555504,
      "start_brk": 94188243599360,
      "arg_start": 140725059845923,
      "arg_end": 140725059845934,
      "env_start": 140725059845934,
      "env_end": 140725059846125,
      "exit_code": 0,
      "state_pretty": "Sleeping in an interruptible wait"
    }

    $ cat /proc/1/stat | jc --proc-pid-stat -p -r
    {
      "pid": 1,
      "comm": "systemd",
      "state": "S",
      "ppid": 0,
      "pgrp": 1,
      "session": 1,
      "tty_nr": 0,
      "tpg_id": -1,
      "flags": 4194560,
      "minflt": 23478,
      "cminflt": 350218,
      "majflt": 99,
      "cmajflt": 472,
      "utime": 107,
      "stime": 461,
      "cutime": 2672,
      "cstime": 4402,
      "priority": 20,
      "nice": 0,
      "num_threads": 1,
      "itrealvalue": 0,
      "starttime": 128,
      "vsize": 174063616,
      "rss": 3313,
      "rsslim": 18446744073709551615,
      "startcode": 94188219072512,
      "endcode": 94188219899461,
      "startstack": 140725059845296,
      "kstkeep": 0,
      "kstkeip": 0,
      "signal": 0,
      "blocked": 671173123,
      "sigignore": 4096,
      "sigcatch": 1260,
      "wchan": 1,
      "nswap": 0,
      "cnswap": 0,
      "exit_signal": 17,
      "processor": 0,
      "rt_priority": 0,
      "policy": 0,
      "delayacct_blkio_ticks": 18,
      "guest_time": 0,
      "cguest_time": 0,
      "start_data": 94188220274448,
      "end_data": 94188220555504,
      "start_brk": 94188243599360,
      "arg_start": 140725059845923,
      "arg_end": 140725059845934,
      "env_start": 140725059845934,
      "env_end": 140725059846125,
      "exit_code": 0
    }
"""
import re
from typing import Dict
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.1'
    description = '`/proc/<pid>/stat` file parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux']
    tags = ['file']
    hidden = True
__version__ = info.version

def _process(proc_data: Dict) -> Dict:
    if False:
        return 10
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (Dictionary) raw structured data to process\n\n    Returns:\n\n        Dictionary. Structured to conform to the schema.\n    '
    state_map = {'R': 'Running', 'S': 'Sleeping in an interruptible wait', 'D': 'Waiting in uninterruptible disk sleep', 'Z': 'Zombie', 'T': 'Stopped (on a signal) or trace stopped', 't': 'Tracing stop', 'W': 'Paging', 'X': 'Dead', 'x': 'Dead', 'K': 'Wakekill', 'W': 'Waking', 'P': 'Parked'}
    if 'state' in proc_data:
        proc_data['state_pretty'] = state_map[proc_data['state']]
    for (key, val) in proc_data.items():
        try:
            proc_data[key] = int(val)
        except Exception:
            pass
    return proc_data

def parse(data: str, raw: bool=False, quiet: bool=False) -> Dict:
    if False:
        i = 10
        return i + 15
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        Dictionary. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output: Dict = {}
    if jc.utils.has_data(data):
        line_re = re.compile('\n            ^(?P<pid>\\d+)\\s\n            \\((?P<comm>.+)\\)\\s\n            (?P<state>\\S)\\s\n            (?P<ppid>\\d+)\\s\n            (?P<pgrp>\\d+)\\s\n            (?P<session>\\d+)\\s\n            (?P<tty_nr>\\d+)\\s\n            (?P<tpg_id>-?\\d+)\\s\n            (?P<flags>\\d+)\\s\n            (?P<minflt>\\d+)\\s\n            (?P<cminflt>\\d+)\\s\n            (?P<majflt>\\d+)\\s\n            (?P<cmajflt>\\d+)\\s\n            (?P<utime>\\d+)\\s\n            (?P<stime>\\d+)\\s\n            (?P<cutime>\\d+)\\s\n            (?P<cstime>\\d+)\\s\n            (?P<priority>\\d+)\\s\n            (?P<nice>\\d+)\\s\n            (?P<num_threads>\\d+)\\s\n            (?P<itrealvalue>\\d+)\\s\n            (?P<starttime>\\d+)\\s\n            (?P<vsize>\\d+)\\s\n            (?P<rss>\\d+)\\s\n            (?P<rsslim>\\d+)\\s\n            (?P<startcode>\\d+)\\s\n            (?P<endcode>\\d+)\\s\n            (?P<startstack>\\d+)\\s\n            (?P<kstkeep>\\d+)\\s\n            (?P<kstkeip>\\d+)\\s\n            (?P<signal>\\d+)\\s\n            (?P<blocked>\\d+)\\s\n            (?P<sigignore>\\d+)\\s\n            (?P<sigcatch>\\d+)\\s\n            (?P<wchan>\\d+)\\s\n            (?P<nswap>\\d+)\\s\n            (?P<cnswap>\\d+)\\s\n            (?P<exit_signal>\\d+)\\s\n            (?P<processor>\\d+)\\s\n            (?P<rt_priority>\\d+)\\s\n            (?P<policy>\\d+)\\s\n            (?P<delayacct_blkio_ticks>\\d+)\\s\n            (?P<guest_time>\\d+)\\s\n            (?P<cguest_time>\\d+)\\s\n            (?P<start_data>\\d+)\\s\n            (?P<end_data>\\d+)\\s\n            (?P<start_brk>\\d+)\\s\n            (?P<arg_start>\\d+)\\s\n            (?P<arg_end>\\d+)\\s\n            (?P<env_start>\\d+)\\s\n            (?P<env_end>\\d+)\\s\n            (?P<exit_code>\\d+)\n        ', re.VERBOSE | re.DOTALL)
        line_match = line_re.search(data)
        if line_match:
            raw_output = line_match.groupdict()
    return raw_output if raw else _process(raw_output)