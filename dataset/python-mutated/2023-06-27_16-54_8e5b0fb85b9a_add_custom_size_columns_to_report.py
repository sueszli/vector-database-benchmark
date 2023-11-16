"""Add custom size columns to report schedule

Revision ID: 8e5b0fb85b9a
Revises: 6fbe660cac39
Create Date: 2023-06-27 16:54:57.161475

"""
import sqlalchemy as sa
from alembic import op
revision = '8e5b0fb85b9a'
down_revision = '6fbe660cac39'

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('report_schedule', sa.Column('custom_width', sa.Integer(), nullable=True))
    op.add_column('report_schedule', sa.Column('custom_height', sa.Integer(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('report_schedule', 'custom_width')
    op.drop_column('report_schedule', 'custom_height')