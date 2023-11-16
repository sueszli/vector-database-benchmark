"""This example demonstrates a setuptools entry point.

See https://hypothesis.readthedocs.io/en/latest/strategies.html#registering-strategies-via-setuptools-entry-points
for details and documentation.
"""

class MyCustomType:

    def __init__(self, x: int):
        if False:
            print('Hello World!')
        assert x >= 0, f'got {x}, but only positive numbers are allowed'
        self.x = x

def _hypothesis_setup_hook():
    if False:
        print('Hello World!')
    import hypothesis.strategies as st
    st.register_type_strategy(MyCustomType, st.integers(min_value=0).map(MyCustomType))