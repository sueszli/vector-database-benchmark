"""A stable file for which we can write patches.  Don't move stuff around!"""
from pathlib import Path
import hypothesis
import hypothesis.strategies as st
WHERE_TOP = Path(__file__).relative_to(Path.cwd())

@hypothesis.given(st.integers())
def fn_top(x):
    if False:
        print('Hello World!')
    'A trivial test function.'