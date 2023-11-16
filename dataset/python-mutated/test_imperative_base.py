import contextlib
from paddle import base

@contextlib.contextmanager
def new_program_scope(main=None, startup=None, scope=None):
    if False:
        while True:
            i = 10
    prog = main if main else base.Program()
    startup_prog = startup if startup else base.Program()
    scope = scope if scope else base.core.Scope()
    with base.scope_guard(scope):
        with base.program_guard(prog, startup_prog):
            with base.unique_name.guard():
                yield