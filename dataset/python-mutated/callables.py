"""A stable file for which we can write patches.  Don't move stuff around!"""
from pathlib import Path
from hypothesis import example, given, strategies as st
WHERE = Path(__file__).relative_to(Path.cwd())

@given(st.integers())
def fn(x):
    if False:
        return 10
    'A trivial test function.'

class Cases:

    @example(n=0, label='whatever')
    @given(st.integers(), st.text())
    def mth(self, n, label):
        if False:
            for i in range(10):
                print('nop')
        'Indented method with existing example decorator.'

@given(st.integers())
@example(x=2).via('not a literal when repeated ' * 2)
@example(x=1).via('covering example')
def covered(x):
    if False:
        i = 10
        return i + 15
    'A test function with a removable explicit example.'