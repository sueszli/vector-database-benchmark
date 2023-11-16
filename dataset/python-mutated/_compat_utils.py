import click

def _get_click_major() -> int:
    if False:
        i = 10
        return i + 15
    return int(click.__version__.split('.')[0])