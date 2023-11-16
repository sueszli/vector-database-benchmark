"""add_columns_start_time_and_end_time.

Revision ID: f78059038d01
Revises: 05844c702676
Create Date: 2022-01-25 09:26:35.820814

"""
from dagster._core.storage.migration.utils import add_run_record_start_end_timestamps, drop_run_record_start_end_timestamps
revision = 'f78059038d01'
down_revision = '05844c702676'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    add_run_record_start_end_timestamps()

def downgrade():
    if False:
        i = 10
        return i + 15
    drop_run_record_start_end_timestamps()