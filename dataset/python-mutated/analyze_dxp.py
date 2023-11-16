"""
Some helper functions to analyze the output of sys.getdxp() (which is
only available if Python was built with -DDYNAMIC_EXECUTION_PROFILE).
These will tell you which opcodes have been executed most frequently
in the current process, and, if Python was also built with -DDXPAIRS,
will tell you which instruction _pairs_ were executed most frequently,
which may help in choosing new instructions.

If Python was built without -DDYNAMIC_EXECUTION_PROFILE, importing
this module will raise a RuntimeError.

If you're running a script you want to profile, a simple way to get
the common pairs is:

$ PYTHONPATH=$PYTHONPATH:<python_srcdir>/Tools/scripts ./python -i -O the_script.py --args
...
> from analyze_dxp import *
> s = render_common_pairs()
> open('/tmp/some_file', 'w').write(s)
"""
import copy
import opcode
import operator
import sys
import threading
if not hasattr(sys, 'getdxp'):
    raise RuntimeError("Can't import analyze_dxp: Python built without -DDYNAMIC_EXECUTION_PROFILE.")
_profile_lock = threading.RLock()
_cumulative_profile = sys.getdxp()

def has_pairs(profile):
    if False:
        print('Hello World!')
    'Returns True if the Python that produced the argument profile\n    was built with -DDXPAIRS.'
    return len(profile) > 0 and isinstance(profile[0], list)

def reset_profile():
    if False:
        return 10
    'Forgets any execution profile that has been gathered so far.'
    with _profile_lock:
        sys.getdxp()
        global _cumulative_profile
        _cumulative_profile = sys.getdxp()

def merge_profile():
    if False:
        print('Hello World!')
    "Reads sys.getdxp() and merges it into this module's cached copy.\n\n    We need this because sys.getdxp() 0s itself every time it's called."
    with _profile_lock:
        new_profile = sys.getdxp()
        if has_pairs(new_profile):
            for first_inst in range(len(_cumulative_profile)):
                for second_inst in range(len(_cumulative_profile[first_inst])):
                    _cumulative_profile[first_inst][second_inst] += new_profile[first_inst][second_inst]
        else:
            for inst in range(len(_cumulative_profile)):
                _cumulative_profile[inst] += new_profile[inst]

def snapshot_profile():
    if False:
        for i in range(10):
            print('nop')
    'Returns the cumulative execution profile until this call.'
    with _profile_lock:
        merge_profile()
        return copy.deepcopy(_cumulative_profile)

def common_instructions(profile):
    if False:
        print('Hello World!')
    'Returns the most common opcodes in order of descending frequency.\n\n    The result is a list of tuples of the form\n      (opcode, opname, # of occurrences)\n\n    '
    if has_pairs(profile) and profile:
        inst_list = profile[-1]
    else:
        inst_list = profile
    result = [(op, opcode.opname[op], count) for (op, count) in enumerate(inst_list) if count > 0]
    result.sort(key=operator.itemgetter(2), reverse=True)
    return result

def common_pairs(profile):
    if False:
        return 10
    'Returns the most common opcode pairs in order of descending frequency.\n\n    The result is a list of tuples of the form\n      ((1st opcode, 2nd opcode),\n       (1st opname, 2nd opname),\n       # of occurrences of the pair)\n\n    '
    if not has_pairs(profile):
        return []
    result = [((op1, op2), (opcode.opname[op1], opcode.opname[op2]), count) for (op1, op1profile) in enumerate(profile[:-1]) for (op2, count) in enumerate(op1profile) if count > 0]
    result.sort(key=operator.itemgetter(2), reverse=True)
    return result

def render_common_pairs(profile=None):
    if False:
        while True:
            i = 10
    "Renders the most common opcode pairs to a string in order of\n    descending frequency.\n\n    The result is a series of lines of the form:\n      # of occurrences: ('1st opname', '2nd opname')\n\n    "
    if profile is None:
        profile = snapshot_profile()

    def seq():
        if False:
            for i in range(10):
                print('nop')
        for (_, ops, count) in common_pairs(profile):
            yield ('%s: %s\n' % (count, ops))
    return ''.join(seq())