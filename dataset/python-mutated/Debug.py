"""SCons.Debug

Code for debugging SCons internal things.  Shouldn't be
needed by most users. Quick shortcuts:

from SCons.Debug import caller_trace
caller_trace()

"""
__revision__ = 'src/engine/SCons/Debug.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os
import sys
import time
import weakref
import inspect
track_instances = False
tracked_classes = {}

def logInstanceCreation(instance, name=None):
    if False:
        print('Hello World!')
    if name is None:
        name = instance.__class__.__name__
    if name not in tracked_classes:
        tracked_classes[name] = []
    if hasattr(instance, '__dict__'):
        tracked_classes[name].append(weakref.ref(instance))
    else:
        tracked_classes[name].append(instance)

def string_to_classes(s):
    if False:
        print('Hello World!')
    if s == '*':
        return sorted(tracked_classes.keys())
    else:
        return s.split()

def fetchLoggedInstances(classes='*'):
    if False:
        return 10
    classnames = string_to_classes(classes)
    return [(cn, len(tracked_classes[cn])) for cn in classnames]

def countLoggedInstances(classes, file=sys.stdout):
    if False:
        print('Hello World!')
    for classname in string_to_classes(classes):
        file.write('%s: %d\n' % (classname, len(tracked_classes[classname])))

def listLoggedInstances(classes, file=sys.stdout):
    if False:
        for i in range(10):
            print('nop')
    for classname in string_to_classes(classes):
        file.write('\n%s:\n' % classname)
        for ref in tracked_classes[classname]:
            if inspect.isclass(ref):
                obj = ref()
            else:
                obj = ref
            if obj is not None:
                file.write('    %s\n' % repr(obj))

def dumpLoggedInstances(classes, file=sys.stdout):
    if False:
        i = 10
        return i + 15
    for classname in string_to_classes(classes):
        file.write('\n%s:\n' % classname)
        for ref in tracked_classes[classname]:
            obj = ref()
            if obj is not None:
                file.write('    %s:\n' % obj)
                for (key, value) in obj.__dict__.items():
                    file.write('        %20s : %s\n' % (key, value))
if sys.platform[:5] == 'linux':

    def memory():
        if False:
            i = 10
            return i + 15
        with open('/proc/self/stat') as f:
            mstr = f.read()
        mstr = mstr.split()[22]
        return int(mstr)
elif sys.platform[:6] == 'darwin':

    def memory():
        if False:
            for i in range(10):
                print('nop')
        return 0
else:
    try:
        import resource
    except ImportError:
        try:
            import win32process
            import win32api
        except ImportError:

            def memory():
                if False:
                    return 10
                return 0
        else:

            def memory():
                if False:
                    while True:
                        i = 10
                process_handle = win32api.GetCurrentProcess()
                memory_info = win32process.GetProcessMemoryInfo(process_handle)
                return memory_info['PeakWorkingSetSize']
    else:

        def memory():
            if False:
                print('Hello World!')
            res = resource.getrusage(resource.RUSAGE_SELF)
            return res[4]

def caller_stack():
    if False:
        return 10
    import traceback
    tb = traceback.extract_stack()
    tb = tb[:-2]
    result = []
    for back in tb:
        key = back[:3]
        result.append('%s:%d(%s)' % func_shorten(key))
    return result
caller_bases = {}
caller_dicts = {}

def caller_trace(back=0):
    if False:
        return 10
    '\n    Trace caller stack and save info into global dicts, which\n    are printed automatically at the end of SCons execution.\n    '
    global caller_bases, caller_dicts
    import traceback
    tb = traceback.extract_stack(limit=3 + back)
    tb.reverse()
    callee = tb[1][:3]
    caller_bases[callee] = caller_bases.get(callee, 0) + 1
    for caller in tb[2:]:
        caller = callee + caller[:3]
        try:
            entry = caller_dicts[callee]
        except KeyError:
            caller_dicts[callee] = entry = {}
        entry[caller] = entry.get(caller, 0) + 1
        callee = caller

def _dump_one_caller(key, file, level=0):
    if False:
        print('Hello World!')
    leader = '      ' * level
    for (v, c) in sorted([(-v, c) for (c, v) in caller_dicts[key].items()]):
        file.write('%s  %6d %s:%d(%s)\n' % ((leader, -v) + func_shorten(c[-3:])))
        if c in caller_dicts:
            _dump_one_caller(c, file, level + 1)

def dump_caller_counts(file=sys.stdout):
    if False:
        i = 10
        return i + 15
    for k in sorted(caller_bases.keys()):
        file.write('Callers of %s:%d(%s), %d calls:\n' % (func_shorten(k) + (caller_bases[k],)))
        _dump_one_caller(k, file)
shorten_list = [('/scons/SCons/', 1), ('/src/engine/SCons/', 1), ('/usr/lib/python', 0)]
if os.sep != '/':
    shorten_list = [(t[0].replace('/', os.sep), t[1]) for t in shorten_list]

def func_shorten(func_tuple):
    if False:
        while True:
            i = 10
    f = func_tuple[0]
    for t in shorten_list:
        i = f.find(t[0])
        if i >= 0:
            if t[1]:
                i = i + len(t[0])
            return (f[i:],) + func_tuple[1:]
    return func_tuple
TraceFP = {}
if sys.platform == 'win32':
    TraceDefault = 'con'
else:
    TraceDefault = '/dev/tty'
TimeStampDefault = None
StartTime = time.time()
PreviousTime = StartTime

def Trace(msg, file=None, mode='w', tstamp=None):
    if False:
        while True:
            i = 10
    'Write a trace message to a file.  Whenever a file is specified,\n    it becomes the default for the next call to Trace().'
    global TraceDefault
    global TimeStampDefault
    global PreviousTime
    if file is None:
        file = TraceDefault
    else:
        TraceDefault = file
    if tstamp is None:
        tstamp = TimeStampDefault
    else:
        TimeStampDefault = tstamp
    try:
        fp = TraceFP[file]
    except KeyError:
        try:
            fp = TraceFP[file] = open(file, mode)
        except TypeError:
            fp = file
    if tstamp:
        now = time.time()
        fp.write('%8.4f %8.4f:  ' % (now - StartTime, now - PreviousTime))
        PreviousTime = now
    fp.write(msg)
    fp.flush()
    fp.close()