import alembic.command
import click
from warehouse.cli.db import db

@db.command()
@click.option('--resolve-dependencies', '-r', is_flag=True, help='Treat dependency versions as down revisions')
@click.pass_obj
def heads(config, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Show current available heads.\n    '
    alembic.command.heads(config.alembic_config(), **kwargs)