from __future__ import annotations
import argparse
import os
import stat
from subprocess import CalledProcessError
from subprocess import check_output
import gdb
import pwndbg.commands
from pwndbg.color import message
from pwndbg.commands import CommandCategory
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Attaches to a given pid, process name or device file.\n\nThis command wraps the original GDB `attach` command to add the ability\nto debug a process with given name. In such case the process identifier is\nfetched via the `pidof <name>` command.\n\nOriginal GDB attach command help:\n    Attach to a process or file outside of GDB.\n    This command attaches to another target, of the same type as your last\n    "target" command ("info files" will show your target stack).\n    The command may take as argument a process id or a device file.\n    For a process id, you must have permission to send the process a signal,\n    and it must have the same effective uid as the debugger.\n    When using "attach" with a process id, the debugger finds the\n    program running in the process, looking first in the current working\n    directory, or (if not found there) using the source file search path\n    (see the "directory" command).  You can also use the "file" command\n    to specify the program, and to load its symbol table.')
parser.add_argument('target', type=str, help='pid, process name or device file to attach to')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.START)
def attachp(target) -> None:
    if False:
        for i in range(10):
            print('nop')
    try:
        resolved_target = int(target)
    except ValueError:
        if _is_device(target):
            resolved_target = target
        else:
            try:
                pids = check_output(['pidof', target]).decode().rstrip('\n').split(' ')
            except FileNotFoundError:
                print(message.error('Error: did not find `pidof` command'))
                return
            except CalledProcessError:
                pids = []
            if not pids:
                print(message.error(f'Process {target} not found'))
                return
            if len(pids) > 1:
                print(message.warn(f"Found pids: {', '.join(pids)} (use `attach <pid>`)"))
                return
            resolved_target = int(pids[0])
    print(message.on(f'Attaching to {resolved_target}'))
    try:
        gdb.execute(f'attach {resolved_target}')
    except gdb.error as e:
        print(message.error(f'Error: {e}'))

def _is_device(path) -> bool:
    if False:
        print('Hello World!')
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError:
        return False
    if stat.S_ISCHR(mode):
        return True
    return False