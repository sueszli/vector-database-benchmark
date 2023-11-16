import alembic.command
import click
from warehouse.cli.db import db

@db.command()
@click.argument('revision', required=True)
@click.pass_obj
def downgrade(config, revision, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Revert to a previous version.\n    '
    alembic.command.downgrade(config.alembic_config(), revision, **kwargs)