"""add tick selector index.

Revision ID: 721d858e1dda
Revises: b601eb913efa
Create Date: 2022-03-25 10:28:29.065161

"""
from dagster._core.storage.migration.utils import create_tick_selector_index
revision = '721d858e1dda'
down_revision = 'b601eb913efa'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    create_tick_selector_index()

def downgrade():
    if False:
        while True:
            i = 10
    pass