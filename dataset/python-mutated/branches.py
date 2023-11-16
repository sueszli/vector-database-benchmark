import alembic.command
import click
from warehouse.cli.db import db

@db.command()
@click.pass_obj
def branches(config, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Show current branch points.\n    '
    alembic.command.branches(config.alembic_config(), **kwargs)