"""add run job index

Revision ID: 16689497301f
Revises: 6df03f4b1efb
Create Date: 2023-02-01 11:27:14.146322

"""
from dagster._core.storage.migration.utils import add_run_job_index, drop_run_job_index
revision = '16689497301f'
down_revision = '6df03f4b1efb'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    add_run_job_index()

def downgrade():
    if False:
        i = 10
        return i + 15
    drop_run_job_index()