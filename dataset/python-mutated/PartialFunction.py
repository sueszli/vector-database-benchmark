from functools import partial

def function(value, expected, lower=False):
    if False:
        while True:
            i = 10
    if lower is True:
        value = value.lower()
    assert value == expected
partial_function = partial(function, expected='value')