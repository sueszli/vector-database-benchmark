"""
This coverage plug-in attempts to cover JIT'd functions and methods that were previously missed in code coverage. Any
function and method that was passed through/decorated with torch.jit.script or torch.jit.script_method should now be
marked covered when coverage is run with this plug-in.

DISCLAIMER: note that this will mark the entire JIT'd function/method as covered without seeking proof that the
compiled code has been executed. This means that even if the code chunk is merely compiled and not run, it will get
marked as covered.
"""
from inspect import getsourcefile, getsourcelines, isclass, iscode, isfunction, ismethod, ismodule
from time import time
from typing import Any
from coverage import CoverageData, CoveragePlugin
cov_data = CoverageData(basename=f'.coverage.jit.{time()}')

def is_not_builtin_class(obj: Any) -> bool:
    if False:
        while True:
            i = 10
    return isclass(obj) and (not type(obj).__module__ == 'builtins')

class JitPlugin(CoveragePlugin):
    """
    dynamic_context is an overridden function that gives us access to every frame run during the coverage process. We
    look for when the function being run is `should_drop`, as all functions that get passed into `should_drop` will be
    compiled and thus should be marked as covered.
    """

    def dynamic_context(self, frame: Any) -> None:
        if False:
            i = 10
            return i + 15
        if frame.f_code.co_name == 'should_drop':
            obj = frame.f_locals['fn']
            if is_not_builtin_class(obj) or ismodule(obj) or ismethod(obj) or isfunction(obj) or iscode(obj):
                filename = getsourcefile(obj)
                if filename:
                    try:
                        (sourcelines, starting_lineno) = getsourcelines(obj)
                    except OSError:
                        pass
                    else:
                        line_data = {filename: range(starting_lineno, starting_lineno + len(sourcelines))}
                        cov_data.add_lines(line_data)
        super().dynamic_context(frame)

def coverage_init(reg: Any, options: Any) -> None:
    if False:
        print('Hello World!')
    reg.add_dynamic_context(JitPlugin())