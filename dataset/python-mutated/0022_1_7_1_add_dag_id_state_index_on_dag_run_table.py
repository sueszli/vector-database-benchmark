"""Add ``dag_id``/``state`` index on ``dag_run`` table

Revision ID: 127d2bf2dfa7
Revises: 5e7d17757c7a
Create Date: 2017-01-25 11:43:51.635667

"""
from __future__ import annotations
from alembic import op
revision = '127d2bf2dfa7'
down_revision = '5e7d17757c7a'
branch_labels = None
depends_on = None
airflow_version = '1.7.1.3'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_index('dag_id_state', 'dag_run', ['dag_id', 'state'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index('dag_id_state', table_name='dag_run')