"""Add custom_operator_name column

Revision ID: 788397e78828
Revises: 937cbd173ca1
Create Date: 2023-06-12 10:46:52.125149

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '788397e78828'
down_revision = '937cbd173ca1'
branch_labels = None
depends_on = None
airflow_version = '2.7.0'
TABLE_NAME = 'task_instance'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Apply Add custom_operator_name column'
    with op.batch_alter_table(TABLE_NAME) as batch_op:
        batch_op.add_column(sa.Column('custom_operator_name', sa.VARCHAR(length=1000), nullable=True))

def downgrade():
    if False:
        return 10
    'Unapply Add custom_operator_name column'
    with op.batch_alter_table(TABLE_NAME) as batch_op:
        batch_op.drop_column('custom_operator_name')