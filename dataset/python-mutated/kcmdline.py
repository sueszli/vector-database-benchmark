from __future__ import annotations
import argparse
import pwndbg.commands
import pwndbg.gdblib.kernel
from pwndbg.commands import CommandCategory
parser = argparse.ArgumentParser(description='Return the kernel commandline (/proc/cmdline).')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.KERNEL)
@pwndbg.commands.OnlyWhenQemuKernel
@pwndbg.commands.OnlyWithKernelDebugSyms
@pwndbg.commands.OnlyWhenPagingEnabled
def kcmdline() -> None:
    if False:
        i = 10
        return i + 15
    print(pwndbg.gdblib.kernel.kcmdline())