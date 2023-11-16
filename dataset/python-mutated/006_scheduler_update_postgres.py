"""scheduler update.

Revision ID: 5c18fd3c2957
Revises: c39c047fa021
Create Date: 2020-06-10 10:00:57.793622

"""
from alembic import op
from dagster._core.storage.migration.utils import get_currently_upgrading_instance, has_table
from sqlalchemy import inspect
revision = '5c18fd3c2957'
down_revision = 'c39c047fa021'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    inspector = inspect(op.get_bind())
    if 'postgresql' not in inspector.dialect.dialect_description:
        return
    instance = get_currently_upgrading_instance()
    if instance.scheduler:
        instance.scheduler.wipe(instance)
    if has_table('schedule_ticks'):
        op.drop_table('schedule_ticks')

def downgrade():
    if False:
        i = 10
        return i + 15
    pass