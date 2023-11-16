"""add range index.

Revision ID: 17154c80d885
Revises: f4eed4c26e2c
Create Date: 2022-01-20 11:45:26.092743

"""
from dagster._core.storage.migration.utils import create_run_range_indices
revision = '17154c80d885'
down_revision = 'f4eed4c26e2c'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    create_run_range_indices()

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass