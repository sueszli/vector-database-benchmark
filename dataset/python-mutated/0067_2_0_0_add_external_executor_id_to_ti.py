"""Add external executor ID to TI

Revision ID: e1a11ece99cc
Revises: b247b1e3d1ed
Create Date: 2020-09-12 08:23:45.698865

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'e1a11ece99cc'
down_revision = 'b247b1e3d1ed'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        return 10
    'Apply Add external executor ID to TI'
    with op.batch_alter_table('task_instance', schema=None) as batch_op:
        batch_op.add_column(sa.Column('external_executor_id', sa.String(length=250), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Unapply Add external executor ID to TI'
    with op.batch_alter_table('task_instance', schema=None) as batch_op:
        batch_op.drop_column('external_executor_id')