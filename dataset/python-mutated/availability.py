import os
from .compilation import compile_run_strings
from .util import CompilerNotFoundError

def has_fortran():
    if False:
        return 10
    if not hasattr(has_fortran, 'result'):
        try:
            ((stdout, stderr), info) = compile_run_strings([('main.f90', 'program foo\nprint *, "hello world"\nend program')], clean=True)
        except CompilerNotFoundError:
            has_fortran.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError('Failed to compile test program:\n%s\n%s\n' % (stdout, stderr))
                has_fortran.result = False
            else:
                has_fortran.result = True
    return has_fortran.result

def has_c():
    if False:
        i = 10
        return i + 15
    if not hasattr(has_c, 'result'):
        try:
            ((stdout, stderr), info) = compile_run_strings([('main.c', '#include <stdio.h>\nint main(){\nprintf("hello world\\n");\nreturn 0;\n}')], clean=True)
        except CompilerNotFoundError:
            has_c.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError('Failed to compile test program:\n%s\n%s\n' % (stdout, stderr))
                has_c.result = False
            else:
                has_c.result = True
    return has_c.result

def has_cxx():
    if False:
        while True:
            i = 10
    if not hasattr(has_cxx, 'result'):
        try:
            ((stdout, stderr), info) = compile_run_strings([('main.cxx', '#include <iostream>\nint main(){\nstd::cout << "hello world" << std::endl;\n}')], clean=True)
        except CompilerNotFoundError:
            has_cxx.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError('Failed to compile test program:\n%s\n%s\n' % (stdout, stderr))
                has_cxx.result = False
            else:
                has_cxx.result = True
    return has_cxx.result