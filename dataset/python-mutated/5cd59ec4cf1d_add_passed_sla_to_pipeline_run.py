"""Add passed_sla to pipeline run

Revision ID: 5cd59ec4cf1d
Revises: 84de4cdd6126
Create Date: 2022-10-21 11:39:38.166335

"""
from alembic import op
import sqlalchemy as sa
revision = '5cd59ec4cf1d'
down_revision = '84de4cdd6126'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.add_column('pipeline_run', sa.Column('passed_sla', sa.Boolean(), default=False))

def downgrade() -> None:
    if False:
        return 10
    op.drop_column('pipeline_run', 'passed_sla')