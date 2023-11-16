"""Add metrics to block_runs

Revision ID: ec5df57a1c60
Revises: 8971d4cd5b39
Create Date: 2022-12-04 22:18:55.232804

"""
from alembic import op
import sqlalchemy as sa
revision = 'ec5df57a1c60'
down_revision = '8971d4cd5b39'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.add_column('block_run', sa.Column('metrics', sa.JSON(), nullable=True))

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_column('block_run', 'metrics')