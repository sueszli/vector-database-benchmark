import os
import re
import shlex
import subprocess
import sys
from hypothesistooling import ROOT

def print_command(command, args):
    if False:
        for i in range(10):
            print('nop')
    args = list(args)
    ranges = []
    for (i, v) in enumerate(args):
        if os.path.exists(v):
            if not ranges or ranges[-1][-1] < i - 1:
                ranges.append([i, i])
            elif ranges[-1][-1] + 1 == i:
                ranges[-1][-1] += 1
    for (i, j) in ranges:
        if j > i:
            args[i] = '...'
            for k in range(i + 1, j + 1):
                args[k] = None
    args = [v for v in args if v is not None]
    print(command, *map(shlex.quote, args))

def run_script(script, *args, **kwargs):
    if False:
        print('Hello World!')
    print_command(script, args)
    return subprocess.check_call([os.path.join(SCRIPTS, script), *args], **kwargs)
SCRIPTS = ROOT / 'tooling' / 'scripts'
COMMON = SCRIPTS / 'common.sh'

def __calc_script_variables():
    if False:
        while True:
            i = 10
    exports = re.compile('^export ([A-Z_]+)(=|$)', flags=re.MULTILINE)
    common = COMMON.read_text(encoding='utf-8')
    for (name, _) in exports.findall(common):
        globals()[name] = os.environ[name]
__calc_script_variables()

def tool_path(name):
    if False:
        print('Hello World!')
    return os.path.join(os.path.dirname(sys.executable), name)

def pip_tool(name, *args, **kwargs):
    if False:
        return 10
    print_command(name, args)
    r = subprocess.call([tool_path(name), *args], **kwargs)
    if r != 0:
        sys.exit(r)