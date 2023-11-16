"""add cached status data column.

Revision ID: 6df03f4b1efb
Revises: 958a9495162d
Create Date: 2022-11-16 15:23:53.522887

"""
from dagster._core.storage.migration.utils import add_cached_status_data_column
revision = '6df03f4b1efb'
down_revision = '958a9495162d'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    add_cached_status_data_column()

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass