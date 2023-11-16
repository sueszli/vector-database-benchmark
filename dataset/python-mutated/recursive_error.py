"""

Demonstrates Rich tracebacks for recursion errors.

Rich can exclude frames in the middle to avoid huge tracebacks.

"""
from rich.console import Console

def foo(n):
    if False:
        i = 10
        return i + 15
    return bar(n)

def bar(n):
    if False:
        print('Hello World!')
    return foo(n)
console = Console()
try:
    foo(1)
except Exception:
    console.print_exception(max_frames=20)