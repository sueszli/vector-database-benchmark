"""add_report_format_to_report_schedule_model.py

Revision ID: 19e978e1b9c3
Revises: fc3a3a8ff221
Create Date: 2021-04-06 21:39:52.259223

"""
revision = '19e978e1b9c3'
down_revision = 'fc3a3a8ff221'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('report_schedule', sa.Column('report_format', sa.String(length=50), server_default='PNG', nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('report_schedule', 'report_format')