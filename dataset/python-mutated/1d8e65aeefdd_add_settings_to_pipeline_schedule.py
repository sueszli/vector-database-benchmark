"""Add settings to pipeline schedule

Revision ID: 1d8e65aeefdd
Revises: 26305e46df52
Create Date: 2023-01-26 14:54:26.688979

"""
from alembic import op
import sqlalchemy as sa
revision = '1d8e65aeefdd'
down_revision = '26305e46df52'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    op.add_column('pipeline_schedule', sa.Column('settings', sa.JSON(), nullable=True))

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_column('pipeline_schedule', 'settings')