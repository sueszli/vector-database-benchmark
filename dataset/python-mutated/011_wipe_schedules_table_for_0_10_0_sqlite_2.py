"""wipe schedules table for 0.10.0.

Revision ID: 140198fdfe65
Revises: b22f16781a7c
Create Date: 2021-01-11 22:16:50.896040

"""
from alembic import op
from dagster._core.storage.migration.utils import get_currently_upgrading_instance, has_table
from sqlalchemy import inspect
revision = '140198fdfe65'
down_revision = 'b22f16781a7c'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    inspector = inspect(op.get_bind())
    if 'sqlite' not in inspector.dialect.dialect_description:
        return
    instance = get_currently_upgrading_instance()
    if instance.scheduler:
        instance.scheduler.wipe(instance)
    if has_table('schedule_ticks'):
        op.drop_table('schedule_ticks')

def downgrade():
    if False:
        while True:
            i = 10
    pass