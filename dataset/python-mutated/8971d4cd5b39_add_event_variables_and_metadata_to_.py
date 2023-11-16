"""add event_variables and metadata to pipeline_runs

Revision ID: 8971d4cd5b39
Revises: 2266370f589b
Create Date: 2022-12-01 00:07:24.890239

"""
from alembic import op
import sqlalchemy as sa
revision = '8971d4cd5b39'
down_revision = '2266370f589b'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.add_column('pipeline_run', sa.Column('event_variables', sa.JSON(), nullable=True))
    op.add_column('pipeline_run', sa.Column('metrics', sa.JSON(), nullable=True))

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_column('pipeline_run', 'event_variables')
    op.drop_column('pipeline_run', 'metrics')