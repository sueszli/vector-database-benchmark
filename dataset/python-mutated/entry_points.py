"""Run all functions registered for the "hypothesis" entry point.

This can be used with `st.register_type_strategy` to register strategies for your
custom types, running the relevant code when *hypothesis* is imported instead of
your package.
"""
import importlib.metadata
import os

def get_entry_points():
    if False:
        for i in range(10):
            print('nop')
    try:
        eps = importlib.metadata.entry_points(group='hypothesis')
    except TypeError:
        eps = importlib.metadata.entry_points().get('hypothesis', [])
    yield from eps

def run():
    if False:
        while True:
            i = 10
    if not os.environ.get('HYPOTHESIS_NO_PLUGINS'):
        for entry in get_entry_points():
            hook = entry.load()
            if callable(hook):
                hook()