import textwrap
from typing import Any
import gdb

class DisableBreakpoints:
    """
    Context-manager to temporarily disable all gdb breakpoints, useful if
    there is a risk to hit one during the evaluation of one of our custom
    commands
    """

    def __enter__(self) -> None:
        if False:
            print('Hello World!')
        self.disabled_breakpoints = []
        for b in gdb.breakpoints():
            if b.enabled:
                b.enabled = False
                self.disabled_breakpoints.append(b)

    def __exit__(self, etype: Any, evalue: Any, tb: Any) -> None:
        if False:
            return 10
        for b in self.disabled_breakpoints:
            b.enabled = True

class TensorRepr(gdb.Command):
    """
    Print a human readable representation of the given at::Tensor.
    Usage: torch-tensor-repr EXP

    at::Tensor instances do not have a C++ implementation of a repr method: in
    pytorch, this is done by pure-Python code. As such, torch-tensor-repr
    internally creates a Python wrapper for the given tensor and call repr()
    on it.
    """
    __doc__ = textwrap.dedent(__doc__).strip()

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        gdb.Command.__init__(self, 'torch-tensor-repr', gdb.COMMAND_USER, gdb.COMPLETE_EXPRESSION)

    def invoke(self, args: str, from_tty: bool) -> None:
        if False:
            i = 10
            return i + 15
        args = gdb.string_to_argv(args)
        if len(args) != 1:
            print('Usage: torch-tensor-repr EXP')
            return
        name = args[0]
        with DisableBreakpoints():
            res = gdb.parse_and_eval(f'torch::gdb::tensor_repr({name})')
            print(f'Python-level repr of {name}:')
            print(res.string())
            gdb.parse_and_eval(f'(void)free({int(res)})')
TensorRepr()