"""empty message.

Revision ID: b22f16781a7c
Revises: b32a4f3036d2
Create Date: 2020-06-10 09:05:47.963960

"""
from alembic import op
from dagster._core.storage.migration.utils import get_currently_upgrading_instance, has_table
from sqlalchemy import inspect
revision = 'b22f16781a7c'
down_revision = 'b32a4f3036d2'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
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