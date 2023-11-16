"""Benchmark the time it takes to compile a reflex app."""
import importlib
import reflex
rx = reflex

class State(rx.State):
    """A simple state class with a count variable."""
    count: int = 0

    def increment(self):
        if False:
            return 10
        'Increment the count.'
        self.count += 1

    def decrement(self):
        if False:
            i = 10
            return i + 15
        'Decrement the count.'
        self.count -= 1

class SliderVariation(State):
    """A simple state class with a count variable."""
    value: int = 50

    def set_end(self, value: int):
        if False:
            return 10
        'Increment the count.\n\n        Args:\n            value: The value of the slider.\n        '
        self.value = value

def sample_small_page() -> rx.Component:
    if False:
        return 10
    'A simple page with a button that increments the count.\n\n    Returns:\n        A reflex component.\n    '
    return rx.vstack(*[rx.button(State.count, font_size='2em') for i in range(100)], spacing='1em')

def sample_large_page() -> rx.Component:
    if False:
        i = 10
        return i + 15
    'A large page with a slider that increments the count.\n\n    Returns:\n        A reflex component.\n    '
    return rx.vstack(*[rx.vstack(rx.heading(SliderVariation.value), rx.slider(on_change_end=SliderVariation.set_end), width='100%') for i in range(100)], spacing='1em')

def add_small_pages(app: rx.App):
    if False:
        return 10
    'Add 10 small pages to the app.\n\n    Args:\n        app: The reflex app to add the pages to.\n    '
    for i in range(10):
        app.add_page(sample_small_page, route=f'/{i}')

def add_large_pages(app: rx.App):
    if False:
        print('Hello World!')
    'Add 10 large pages to the app.\n\n    Args:\n        app: The reflex app to add the pages to.\n    '
    for i in range(10):
        app.add_page(sample_large_page, route=f'/{i}')

def test_mean_import_time(benchmark):
    if False:
        print('Hello World!')
    'Test that the mean import time is less than 1 second.\n\n    Args:\n        benchmark: The benchmark fixture.\n    '

    def import_reflex():
        if False:
            print('Hello World!')
        importlib.reload(reflex)
    benchmark(import_reflex)

def test_mean_add_small_page_time(benchmark):
    if False:
        i = 10
        return i + 15
    'Test that the mean add page time is less than 1 second.\n\n    Args:\n        benchmark: The benchmark fixture.\n    '
    app = rx.App(state=State)
    benchmark(add_small_pages, app)

def test_mean_add_large_page_time(benchmark):
    if False:
        for i in range(10):
            print('nop')
    'Test that the mean add page time is less than 1 second.\n\n    Args:\n        benchmark: The benchmark fixture.\n    '
    app = rx.App(state=State)
    results = benchmark(add_large_pages, app)
    print(results)