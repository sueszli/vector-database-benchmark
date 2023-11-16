"""Merge migrations Heads.

Revision ID: 03bc53e68815
Revises: 0a2a5b66e19d, bf00311e1990
Create Date: 2018-11-24 20:21:46.605414

"""
from __future__ import annotations
from alembic import op
revision = '03bc53e68815'
down_revision = ('0a2a5b66e19d', 'bf00311e1990')
branch_labels = None
depends_on = None
airflow_version = '1.10.2'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_index('sm_dag', 'sla_miss', ['dag_id'], unique=False)

def downgrade():
    if False:
        return 10
    op.drop_index('sm_dag', table_name='sla_miss')