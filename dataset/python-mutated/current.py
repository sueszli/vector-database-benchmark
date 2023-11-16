import alembic.command
import click
from warehouse.cli.db import db

@db.command()
@click.pass_obj
def current(config, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Display the current revision for a database.\n    '
    alembic.command.current(config.alembic_config(), **kwargs)