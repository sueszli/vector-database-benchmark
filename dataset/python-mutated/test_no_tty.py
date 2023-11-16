from __future__ import annotations
import sys

def test_no_tty():
    if False:
        while True:
            i = 10
    assert not sys.stdin.isatty()
    assert not sys.stdout.isatty()
    assert not sys.stderr.isatty()