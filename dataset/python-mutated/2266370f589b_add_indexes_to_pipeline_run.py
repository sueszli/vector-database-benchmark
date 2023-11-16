"""Add indexes to pipeline run.

Revision ID: 2266370f589b
Revises: 5cd59ec4cf1d
Create Date: 2022-11-01 18:21:53.877930

"""
from alembic import op
import sqlalchemy as sa
revision = '2266370f589b'
down_revision = '5cd59ec4cf1d'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.create_index(op.f('ix_pipeline_run_execution_date'), 'pipeline_run', ['execution_date'], unique=False)
    op.create_index(op.f('ix_pipeline_run_pipeline_uuid'), 'pipeline_run', ['pipeline_uuid'], unique=False)
    op.create_index(op.f('ix_pipeline_run_status'), 'pipeline_run', ['status'], unique=False)

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_index(op.f('ix_pipeline_run_status'), table_name='pipeline_run')
    op.drop_index(op.f('ix_pipeline_run_pipeline_uuid'), table_name='pipeline_run')
    op.drop_index(op.f('ix_pipeline_run_execution_date'), table_name='pipeline_run')