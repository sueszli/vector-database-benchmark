from __future__ import annotations
import os
import tempfile
import textwrap
from functools import lru_cache
if os.environ.get('TORCHINDUCTOR_WRITE_MISSING_OPS') == '1':

    @lru_cache(None)
    def _record_missing_op(target):
        if False:
            while True:
                i = 10
        with open(f'{tempfile.gettempdir()}/missing_ops.txt', 'a') as fd:
            fd.write(str(target) + '\n')
else:

    def _record_missing_op(target):
        if False:
            while True:
                i = 10
        pass

class OperatorIssue(RuntimeError):

    @staticmethod
    def operator_str(target, args, kwargs):
        if False:
            while True:
                i = 10
        lines = [f'target: {target}'] + [f'args[{i}]: {arg}' for (i, arg) in enumerate(args)]
        if kwargs:
            lines.append(f'kwargs: {kwargs}')
        return textwrap.indent('\n'.join(lines), '  ')

class MissingOperatorWithoutDecomp(OperatorIssue):

    def __init__(self, target, args, kwargs):
        if False:
            return 10
        _record_missing_op(target)
        super().__init__(f'missing lowering\n{self.operator_str(target, args, kwargs)}')

class MissingOperatorWithDecomp(OperatorIssue):

    def __init__(self, target, args, kwargs):
        if False:
            while True:
                i = 10
        _record_missing_op(target)
        super().__init__(f'missing decomposition\n{self.operator_str(target, args, kwargs)}' + textwrap.dedent(f'\n\n                There is a decomposition available for {target} in\n                torch._decomp.get_decompositions().  Please add this operator to the\n                `decompositions` list in torch._inductor.decompositions\n                '))

class LoweringException(OperatorIssue):

    def __init__(self, exc: Exception, target, args, kwargs):
        if False:
            print('Hello World!')
        super().__init__(f'{type(exc).__name__}: {exc}\n{self.operator_str(target, args, kwargs)}')

class InvalidCxxCompiler(RuntimeError):

    def __init__(self):
        if False:
            print('Hello World!')
        from . import config
        super().__init__(f'No working C++ compiler found in {config.__name__}.cpp.cxx: {config.cpp.cxx}')

class CppWrapperCodeGenError(RuntimeError):

    def __init__(self, msg: str):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(f'C++ wrapper codegen error: {msg}')

class CppCompileError(RuntimeError):

    def __init__(self, cmd: list[str], output: str):
        if False:
            return 10
        if isinstance(output, bytes):
            output = output.decode('utf-8')
        super().__init__(textwrap.dedent('\n                    C++ compile error\n\n                    Command:\n                    {cmd}\n\n                    Output:\n                    {output}\n                ').strip().format(cmd=' '.join(cmd), output=output))

class CUDACompileError(CppCompileError):
    pass