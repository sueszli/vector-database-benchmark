"""An ulimit implementation for xonsh."""
import resource
from xonsh.tools import print_exception
_UL_RES = {'c': (getattr(resource, 'RLIMIT_CORE', None), '--core-size', 'the maximum size of core files created.\n                    By setting this limit to zero, core dumps can be disabled.', 'core file size', 'blocks'), 'd': (getattr(resource, 'RLIMIT_DATA', None), '--data-size', "the maximum size of a process' data segment", 'data seg size', 'kbytes'), 'e': (getattr(resource, 'RLIMIT_NICE', None), '--nice', "the maximum scheduling priority (`nice')", 'scheduling priority', ''), 'f': (getattr(resource, 'RLIMIT_FSIZE', None), '--file-size', 'the maximum size of files created by the shell and its children', 'file size', 'blocks'), 'i': (getattr(resource, 'RLIMIT_SIGPENDING', None), '--signals-pending', 'the maximum number of pending signals', 'pending signals', ''), 'l': (getattr(resource, 'RLIMIT_MEMLOCK', None), '--lock-size', 'the maximum size a process may lock into memory', 'max locked memory', 'kbytes'), 'm': (getattr(resource, 'RLIMIT_RSS', None), '--resident-set-size', 'the maximum resident set size.', 'max RSS memory', 'kbytes'), 'n': (getattr(resource, 'RLIMIT_NOFILE', None), '--file-descriptor-count', 'the maximum number of open file descriptors', 'open files', ''), 'q': (getattr(resource, 'RLIMIT_MSGQUEUE', None), '--queue-size', 'the number of bytes that can be allocated for POSIX message queues', 'POSIX message queues', 'bytes'), 'r': (getattr(resource, 'RLIMIT_RTPRIO', None), '--rt-prio', 'the maximum real-time scheduling priority', 'real-time priority', ''), 's': (getattr(resource, 'RLIMIT_STACK', None), '--stack-size', 'the maximum stack size', 'stack size', 'kbytes'), 't': (getattr(resource, 'RLIMIT_CPU', None), '--cpu-time', 'the maximum amount of CPU time in seconds', 'CPU time', 'seconds'), 'u': (getattr(resource, 'RLIMIT_NPROC', None), '--process-count', 'the maximum number of processes available to a single user', 'max user processes', ''), 'v': (getattr(resource, 'RLIMIT_AS', None), '--virtual-memory-size', 'the maximum amount of virtual memory available to the shell', 'virtual memory', 'kbytes')}
_UL_SOFT = 1
_UL_HARD = 2
_UL_BOTH = _UL_SOFT | _UL_HARD

def _ul_set(res, soft=None, hard=None, **kwargs):
    if False:
        return 10
    'Set resource limit'
    if soft == 'unlimited':
        soft = resource.RLIM_INFINITY
    if hard == 'unlimited':
        hard = resource.RLIM_INFINITY
    if soft is None or hard is None or isinstance(soft, str) or isinstance(hard, str):
        (current_soft, current_hard) = resource.getrlimit(res)
        if soft in (None, 'soft'):
            soft = current_soft
        elif soft == 'hard':
            soft = current_hard
        if hard in (None, 'hard'):
            hard = current_hard
        elif hard == 'soft':
            hard = current_soft
    resource.setrlimit(res, (soft, hard))

def _ul_show(res, res_type, desc, unit, opt, long=False, **kwargs):
    if False:
        i = 10
        return i + 15
    'Print out resource limit'
    limit = resource.getrlimit(res)[1 if res_type == _UL_HARD else 0]
    str_limit = 'unlimited' if limit == resource.RLIM_INFINITY else str(limit)
    if long:
        pre = '{:21} {:>14} '.format(desc, '({}{})'.format(unit + ', -' if unit else '-', opt))
    else:
        pre = ''
    print(f'{pre}{str_limit}', file=kwargs['stdout'])

def _ul_add_action(actions, opt, res_type, stderr):
    if False:
        for i in range(10):
            print('nop')
    'Create new and append it to the actions list'
    r = _UL_RES[opt]
    if r[0] is None:
        _ul_unsupported_opt(opt, stderr)
        return False
    actions.append([_ul_show, {'res': r[0], 'res_type': res_type, 'desc': r[3], 'unit': r[4], 'opt': opt}])
    return True

def _ul_add_all_actions(actions, res_type, stderr):
    if False:
        for i in range(10):
            print('nop')
    'Add all supported resources; handles (-a, --all)'
    for k in _UL_RES:
        if _UL_RES[k][0] is None:
            continue
        _ul_add_action(actions, k, res_type, stderr)

def _ul_unknown_opt(arg, stderr):
    if False:
        while True:
            i = 10
    'Print an invalid option message to stderr'
    print(f'ulimit: Invalid option: {arg}', file=stderr, flush=True)
    print("Try 'ulimit --help' for more information", file=stderr, flush=True)

def _ul_unsupported_opt(opt, stderr):
    if False:
        i = 10
        return i + 15
    'Print an unsupported option message to stderr'
    print(f'ulimit: Unsupported option: -{opt}', file=stderr, flush=True)
    print("Try 'ulimit --help' for more information", file=stderr, flush=True)

def _ul_parse_args(args, stderr):
    if False:
        return 10
    'Parse arguments and return a list of actions to be performed'
    if len(args) == 1 and args[0] in ('-h', '--help'):
        return (True, [])
    long_opts = {}
    for k in _UL_RES:
        long_opts[_UL_RES[k][1]] = k
    actions = []
    res_type = _UL_SOFT
    for arg in args:
        if arg in long_opts:
            opt = long_opts[arg]
            if not _ul_add_action(actions, opt, res_type, stderr):
                return (False, [])
        elif arg == '--all':
            _ul_add_all_actions(actions, res_type, stderr)
        elif arg == '--soft':
            res_type = _UL_SOFT
        elif arg == '--hard':
            res_type = _UL_HARD
        elif arg[0] == '-':
            for opt in arg[1:]:
                if opt == 'a':
                    _ul_add_all_actions(actions, res_type, stderr)
                elif opt in _UL_RES:
                    if not _ul_add_action(actions, opt, res_type, stderr):
                        return (False, [])
                elif opt == 'S':
                    res_type = _UL_SOFT
                elif opt == 'H':
                    res_type = _UL_HARD
                else:
                    _ul_unknown_opt(arg, stderr)
                    return (False, [])
        elif arg.isnumeric() or arg in ('unlimited', 'hard', 'soft'):
            if arg.isnumeric():
                limit = int(arg)
            else:
                limit = arg
            if not actions:
                if not _ul_add_action(actions, 'f', res_type, stderr):
                    return (False, [])
            a = actions[-1]
            a[0] = _ul_set
            a[1]['soft'] = limit if _UL_SOFT & res_type else None
            a[1]['hard'] = limit if _UL_HARD & res_type else None
        else:
            _ul_unknown_opt(arg, stderr)
            return (False, [])
    else:
        if not actions:
            if not _ul_add_action(actions, 'f', res_type, stderr):
                return (False, [])
    return (True, actions)

def _ul_show_usage(file):
    if False:
        return 10
    'Print out our help'
    print('Usage: ulimit [-h] [-SH] [-a] [-', end='', file=file)
    print(''.join([k for k in _UL_RES]), end='', file=file)
    print('] [LIMIT]\n', file=file)
    print('Set or get shell resource limits.\n\nProvides control over the resources available to the shell and processes it\ncreates, on systems that allow such control.\n\nOptions:', file=file)
    print('-h, --help\n                    show this help message and exit', file=file)
    print("-S, --soft\n                    use the 'soft' resource limit for the following arguments", file=file)
    print("-H, --hard\n                    use the 'hard' resource limit for the following arguments (default)", file=file)
    print('-a, --all\n                    show all current limits', file=file)
    for k in _UL_RES:
        r = _UL_RES[k]
        opts = f'-{k}, {r[1]}'
        if r[0] is None:
            opts += ' (unsupported)'
        print(f'{opts}\n                    {r[2]}', file=file)
    print("\nNot all options are available on all platforms.\n\nIf LIMIT is given, it is the new value of the specified resource; the special\nLIMIT values `soft', `hard', and `unlimited' stand for the current soft limit,\nthe current hard limit, and no limit, respectively. Otherwise, the current\nvalue of the specified resource is printed. If no option is given, then -f is\nassumed.\n", file=file)

def ulimit(args, stdin, stdout, stderr):
    if False:
        while True:
            i = 10
    'An ulimit implementation'
    (rc, actions) = _ul_parse_args(args, stderr)
    if not rc:
        return 1
    elif not actions:
        _ul_show_usage(stdout)
        return 0
    long = len([a for a in actions if a[0] == _ul_show]) > 1
    try:
        for (fn, args) in actions:
            fn(stdout=stdout, long=long, **args)
        return 0
    except:
        print_exception()
        return 2