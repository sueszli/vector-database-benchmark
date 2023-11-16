"""Add owner_display_name to (Audit) Log table

Revision ID: f7bf2a57d0a6
Revises: 375a816bbbf4
Create Date: 2023-09-12 17:21:45.149658

"""
import sqlalchemy as sa
from alembic import op
revision = 'f7bf2a57d0a6'
down_revision = '375a816bbbf4'
branch_labels = None
depends_on = None
airflow_version = '2.8.0'
TABLE_NAME = 'log'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Adds owner_display_name column to log'
    with op.batch_alter_table(TABLE_NAME) as batch_op:
        batch_op.add_column(sa.Column('owner_display_name', sa.String(500)))

def downgrade():
    if False:
        return 10
    'Removes owner_display_name column from log'
    with op.batch_alter_table(TABLE_NAME) as batch_op:
        batch_op.drop_column('owner_display_name')