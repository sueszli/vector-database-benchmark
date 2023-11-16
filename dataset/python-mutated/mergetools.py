"""Utility functions for managing external merge tools such as kdiff3."""
from __future__ import absolute_import
import os
import shutil
import subprocess
import sys
import tempfile
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nfrom bzrlib import (\n    cmdline,\n    osutils,\n    trace,\n)\n')
known_merge_tools = {'bcompare': 'bcompare {this} {other} {base} {result}', 'kdiff3': 'kdiff3 {base} {this} {other} -o {result}', 'xdiff': 'xxdiff -m -O -M {result} {this} {base} {other}', 'meld': 'meld {base} {this_temp} {other}', 'opendiff': 'opendiff {this} {other} -ancestor {base} -merge {result}', 'winmergeu': 'winmergeu {result}'}

def check_availability(command_line):
    if False:
        for i in range(10):
            print('nop')
    cmd_list = cmdline.split(command_line)
    exe = cmd_list[0]
    if sys.platform == 'win32':
        exe = _get_executable_path(exe)
        if exe is None:
            return False
        (base, ext) = os.path.splitext(exe)
        path_ext = [unicode(s.lower()) for s in os.getenv('PATHEXT', '').split(os.pathsep)]
        return os.path.exists(exe) and ext in path_ext
    else:
        return os.access(exe, os.X_OK) or osutils.find_executable_on_path(exe) is not None

def invoke(command_line, filename, invoker=None):
    if False:
        return 10
    'Invokes the given merge tool command line, substituting the given\n    filename according to the embedded substitution markers. Optionally, it\n    will use the given invoker function instead of the default\n    subprocess_invoker.\n    '
    if invoker is None:
        invoker = subprocess_invoker
    cmd_list = cmdline.split(command_line)
    exe = _get_executable_path(cmd_list[0])
    if exe is not None:
        cmd_list[0] = exe
    (args, tmp_file) = _subst_filename(cmd_list, filename)

    def cleanup(retcode):
        if False:
            for i in range(10):
                print('nop')
        if tmp_file is not None:
            if retcode == 0:
                shutil.move(tmp_file, filename)
            else:
                os.remove(tmp_file)
    return invoker(args[0], args[1:], cleanup)

def _get_executable_path(exe):
    if False:
        while True:
            i = 10
    if os.path.isabs(exe):
        return exe
    return osutils.find_executable_on_path(exe)

def _subst_filename(args, filename):
    if False:
        i = 10
        return i + 15
    subst_names = {'base': filename + u'.BASE', 'this': filename + u'.THIS', 'other': filename + u'.OTHER', 'result': filename}
    tmp_file = None
    subst_args = []
    for arg in args:
        if '{this_temp}' in arg and (not 'this_temp' in subst_names):
            (fh, tmp_file) = tempfile.mkstemp(u'_bzr_mergetools_%s.THIS' % os.path.basename(filename))
            trace.mutter('fh=%r, tmp_file=%r', fh, tmp_file)
            os.close(fh)
            shutil.copy(filename + u'.THIS', tmp_file)
            subst_names['this_temp'] = tmp_file
        arg = _format_arg(arg, subst_names)
        subst_args.append(arg)
    return (subst_args, tmp_file)

def _format_arg(arg, subst_names):
    if False:
        for i in range(10):
            print('nop')
    arg = arg.replace('{base}', subst_names['base'])
    arg = arg.replace('{this}', subst_names['this'])
    arg = arg.replace('{other}', subst_names['other'])
    arg = arg.replace('{result}', subst_names['result'])
    if subst_names.has_key('this_temp'):
        arg = arg.replace('{this_temp}', subst_names['this_temp'])
    return arg

def subprocess_invoker(executable, args, cleanup):
    if False:
        while True:
            i = 10
    retcode = subprocess.call([executable] + args)
    cleanup(retcode)
    return retcode