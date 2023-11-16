"""add range index.

Revision ID: b37316bf5584
Revises: 9c5f00e80ef2
Create Date: 2022-01-20 11:39:54.203976

"""
from dagster._core.storage.migration.utils import create_run_range_indices
revision = 'b37316bf5584'
down_revision = '9c5f00e80ef2'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    create_run_range_indices()

def downgrade():
    if False:
        while True:
            i = 10
    pass