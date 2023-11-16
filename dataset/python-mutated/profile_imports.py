"""A custom importer and regex compiler which logs time spent."""
import sys
import time
import re
_parent_stack = []
_total_stack = {}
_info = {}
_cur_id = 0
_timer = time.time
if sys.platform == 'win32':
    _timer = time.clock

def stack_add(name, frame_name, frame_lineno, scope_name=None):
    if False:
        return 10
    'Start a new record on the stack'
    global _cur_id
    _cur_id += 1
    this_stack = (_cur_id, name)
    if _parent_stack:
        _total_stack[_parent_stack[-1]].append(this_stack)
    _total_stack[this_stack] = []
    _parent_stack.append(this_stack)
    _info[this_stack] = [len(_parent_stack) - 1, frame_name, frame_lineno, scope_name]
    return this_stack

def stack_finish(this, cost):
    if False:
        for i in range(10):
            print('nop')
    'Finish a given entry, and record its cost in time'
    global _parent_stack
    assert _parent_stack[-1] == this, 'import stack does not end with this %s: %s' % (this, _parent_stack)
    _parent_stack.pop()
    _info[this].append(cost)

def log_stack_info(out_file, sorted=True, hide_fast=True):
    if False:
        for i in range(10):
            print('nop')
    out_file.write('%5s %5s %-40s @ %s:%s\n' % ('cum', 'inline', 'name', 'file', 'line'))
    todo = [(value[-1], key) for (key, value) in _info.iteritems() if value[0] == 0]
    if sorted:
        todo.sort()
    while todo:
        (cum_time, cur) = todo.pop()
        children = _total_stack[cur]
        c_times = []
        info = _info[cur]
        if hide_fast and info[-1] < 0.0001:
            continue
        mod_time = info[-1]
        for child in children:
            c_info = _info[child]
            mod_time -= c_info[-1]
            c_times.append((c_info[-1], child))
        out_file.write('%5.1f %5.1f %-40s @ %s:%d\n' % (info[-1] * 1000.0, mod_time * 1000.0, '+' * info[0] + cur[1], info[1], info[2]))
        if sorted:
            c_times.sort()
        else:
            c_times.reverse()
        todo.extend(c_times)
_real_import = __import__

def timed_import(name, globals=None, locals=None, fromlist=None, level=None):
    if False:
        while True:
            i = 10
    'Wrap around standard importer to log import time'
    if globals is None:
        scope_name = None
    else:
        scope_name = globals.get('__name__', None)
        if scope_name is None:
            scope_name = globals.get('__file__', None)
        if scope_name is None:
            scope_name = globals.keys()
        else:
            loc = scope_name.find('bzrlib')
            if loc != -1:
                scope_name = scope_name[loc:]
            loc = scope_name.find('python2.4')
            if loc != -1:
                scope_name = scope_name[loc:]
    frame = sys._getframe(1)
    frame_name = frame.f_globals.get('__name__', '<unknown>')
    extra = ''
    if frame_name.endswith('demandload'):
        extra = '(demandload) '
        frame = sys._getframe(4)
        frame_name = frame.f_globals.get('__name__', '<unknown>')
    elif frame_name.endswith('lazy_import'):
        extra = '[l] '
        frame = sys._getframe(4)
        frame_name = frame.f_globals.get('__name__', '<unknown>')
    if fromlist:
        extra += ' [%s]' % (', '.join(map(str, fromlist)),)
    frame_lineno = frame.f_lineno
    this = stack_add(extra + name, frame_name, frame_lineno, scope_name)
    tstart = _timer()
    try:
        mod = _real_import(name, globals, locals, fromlist)
    finally:
        tload = _timer() - tstart
        stack_finish(this, tload)
    return mod
_real_compile = re._compile

def timed_compile(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    'Log how long it takes to compile a regex'
    frame = sys._getframe(2)
    frame_name = frame.f_globals.get('__name__', '<unknown>')
    extra = ''
    if frame_name.endswith('lazy_regex'):
        extra = '[l] '
        frame = sys._getframe(5)
        frame_name = frame.f_globals.get('__name__', '<unknown>')
    frame_lineno = frame.f_lineno
    this = stack_add(extra + repr(args[0]), frame_name, frame_lineno)
    tstart = _timer()
    try:
        comp = _real_compile(*args, **kwargs)
    finally:
        tcompile = _timer() - tstart
        stack_finish(this, tcompile)
    return comp

def install():
    if False:
        print('Hello World!')
    'Install the hooks for measuring import and regex compile time.'
    __builtins__['__import__'] = timed_import
    re._compile = timed_compile

def uninstall():
    if False:
        while True:
            i = 10
    'Remove the import and regex compile timing hooks.'
    __builtins__['__import__'] = _real_import
    re._compile = _real_compile