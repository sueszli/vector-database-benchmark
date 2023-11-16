"""wipe schedules table for 0.10.0.

Revision ID: b32a4f3036d2
Revises: 375e95bad550
Create Date: 2021-01-11 22:20:01.253271

"""
from alembic import op
from dagster._core.storage.migration.utils import get_currently_upgrading_instance, has_table
from sqlalchemy import inspect
revision = 'b32a4f3036d2'
down_revision = '375e95bad550'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
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
        print('Hello World!')
    pass