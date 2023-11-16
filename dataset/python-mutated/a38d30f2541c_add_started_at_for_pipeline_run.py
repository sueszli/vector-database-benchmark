"""Add started_at for pipeline run

Revision ID: a38d30f2541c
Revises: 386bcfebd48d
Create Date: 2023-08-22 10:54:05.050003

"""
from alembic import op
import sqlalchemy as sa
revision = 'a38d30f2541c'
down_revision = '386bcfebd48d'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    with op.batch_alter_table('pipeline_run', schema=None) as batch_op:
        batch_op.add_column(sa.Column('started_at', sa.DateTime(timezone=True), nullable=True))

def downgrade() -> None:
    if False:
        return 10
    with op.batch_alter_table('pipeline_run', schema=None) as batch_op:
        batch_op.drop_column('started_at')