from typing_extensions import assert_type
import click

@click.group(context_settings={})
def hello() -> None:
    if False:
        i = 10
        return i + 15
    pass
assert_type(hello, click.Group)