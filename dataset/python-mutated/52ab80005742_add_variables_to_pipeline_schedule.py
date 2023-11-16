"""Add variables to pipeline schedule.

Revision ID: 52ab80005742
Revises: c07a23ff782b
Create Date: 2022-08-25 10:31:48.480241

"""
from alembic import op
import sqlalchemy as sa
revision = '52ab80005742'
down_revision = 'c07a23ff782b'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.add_column('pipeline_schedule', sa.Column('variables', sa.JSON(), nullable=True))

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_column('pipeline_schedule', 'variables')