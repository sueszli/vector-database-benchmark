"""Make xcom value column a large binary

Revision ID: bdaa763e6c56
Revises: cc1e65623dc7
Create Date: 2017-08-14 16:06:31.568971

"""
from __future__ import annotations
import dill
import sqlalchemy as sa
from alembic import op
revision = 'bdaa763e6c56'
down_revision = 'cc1e65623dc7'
branch_labels = None
depends_on = None
airflow_version = '1.8.2'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('xcom') as batch_op:
        batch_op.alter_column('value', type_=sa.LargeBinary())

def downgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('xcom') as batch_op:
        batch_op.alter_column('value', type_=sa.PickleType(pickler=dill))