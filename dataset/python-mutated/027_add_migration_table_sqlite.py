"""add migration table.

Revision ID: 54666da3db5c
Revises: 8d66aa722f94
Create Date: 2022-03-23 12:58:43.144576

"""
from dagster._core.storage.migration.utils import create_schedule_secondary_index_table
revision = '54666da3db5c'
down_revision = '8d66aa722f94'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    create_schedule_secondary_index_table()

def downgrade():
    if False:
        i = 10
        return i + 15
    pass