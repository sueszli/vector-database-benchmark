"""Add completed_at.

Revision ID: 053ee2c10d85
Revises: 52ab80005742
Create Date: 2022-08-31 01:22:18.258615

"""
from alembic import op
import sqlalchemy as sa
revision = '053ee2c10d85'
down_revision = '52ab80005742'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.add_column('block_run', sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('pipeline_run', sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('pipeline_run', 'completed_at')
    op.drop_column('block_run', 'completed_at')