from __future__ import annotations
import argparse
import subprocess
import tempfile
import gdb
import pwndbg.commands
import pwndbg.gdblib.vmmap
from pwndbg.commands import CommandCategory
parser = argparse.ArgumentParser(description='ROP gadget search with ropper.', epilog="Example: ropper -- --console; ropper -- --search 'mov e?x'")
parser.add_argument('argument', nargs='*', type=str, help='Arguments to pass to ropper')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.INTEGRATIONS)
@pwndbg.commands.OnlyWithFile
def ropper(argument) -> None:
    if False:
        print('Hello World!')
    with tempfile.NamedTemporaryFile() as corefile:
        if pwndbg.gdblib.proc.alive:
            filename = corefile.name
            gdb.execute(f'gcore {filename}')
        else:
            filename = pwndbg.gdblib.proc.exe
        cmd = ['ropper', '--file', filename]
        cmd += argument
        try:
            io = subprocess.call(cmd)
        except Exception:
            print("Could not run ropper.  Please ensure it's installed and in $PATH.")