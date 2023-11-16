"""Test that runtime typing references are properly attributed to scoped imports."""
from __future__ import annotations
from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from threading import Thread

def fn(thread: Thread):
    if False:
        i = 10
        return i + 15
    from threading import Thread
    x: Thread

def fn(thread: Thread):
    if False:
        while True:
            i = 10
    from threading import Thread
    cast('Thread', thread)

def fn(thread: Thread):
    if False:
        i = 10
        return i + 15
    from threading import Thread
    cast(Thread, thread)