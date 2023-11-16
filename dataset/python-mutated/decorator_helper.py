from paddle import base
__all__ = ['many_times', 'prog_scope']

def many_times(times):
    if False:
        for i in range(10):
            print('nop')

    def __impl__(fn):
        if False:
            return 10

        def __fn__(*args, **kwargs):
            if False:
                print('Hello World!')
            for _ in range(times):
                fn(*args, **kwargs)
        return __fn__
    return __impl__

def prog_scope():
    if False:
        print('Hello World!')

    def __impl__(fn):
        if False:
            print('Hello World!')

        def __fn__(*args, **kwargs):
            if False:
                while True:
                    i = 10
            prog = base.Program()
            startup_prog = base.Program()
            scope = base.core.Scope()
            with base.scope_guard(scope):
                with base.program_guard(prog, startup_prog):
                    fn(*args, **kwargs)
        return __fn__
    return __impl__