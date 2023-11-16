"""add tick selector index.

Revision ID: b601eb913efa
Revises: d32d1d6de793
Create Date: 2022-03-25 10:28:53.372766

"""
from dagster._core.storage.migration.utils import create_tick_selector_index
revision = 'b601eb913efa'
down_revision = 'd32d1d6de793'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    create_tick_selector_index()

def downgrade():
    if False:
        i = 10
        return i + 15
    pass