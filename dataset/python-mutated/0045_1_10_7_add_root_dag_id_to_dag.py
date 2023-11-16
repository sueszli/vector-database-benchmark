"""Add ``root_dag_id`` to ``DAG``

Revision ID: b3b105409875
Revises: d38e04c12aa2
Create Date: 2019-09-28 23:20:01.744775

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.db_types import StringID
revision = 'b3b105409875'
down_revision = 'd38e04c12aa2'
branch_labels = None
depends_on = None
airflow_version = '1.10.7'

def upgrade():
    if False:
        print('Hello World!')
    'Apply Add ``root_dag_id`` to ``DAG``'
    op.add_column('dag', sa.Column('root_dag_id', StringID(), nullable=True))
    op.create_index('idx_root_dag_id', 'dag', ['root_dag_id'], unique=False)

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Add ``root_dag_id`` to ``DAG``'
    op.drop_index('idx_root_dag_id', table_name='dag')
    op.drop_column('dag', 'root_dag_id')