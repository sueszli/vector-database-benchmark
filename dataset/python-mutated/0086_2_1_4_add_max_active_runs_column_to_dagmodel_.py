"""Add ``max_active_runs`` column to ``dag_model`` table

Revision ID: 092435bf5d12
Revises: 97cdd93827b8
Create Date: 2021-09-06 21:29:24.728923

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy import text
revision = '092435bf5d12'
down_revision = '97cdd93827b8'
branch_labels = None
depends_on = None
airflow_version = '2.1.4'

def upgrade():
    if False:
        print('Hello World!')
    'Apply Add ``max_active_runs`` column to ``dag_model`` table'
    op.add_column('dag', sa.Column('max_active_runs', sa.Integer(), nullable=True))
    with op.batch_alter_table('dag_run', schema=None) as batch_op:
        batch_op.create_index('idx_dag_run_dag_id', ['dag_id'])
        batch_op.create_index('idx_dag_run_running_dags', ['state', 'dag_id'], postgresql_where=text("state='running'"), mssql_where=text("state='running'"), sqlite_where=text("state='running'"))

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Add ``max_active_runs`` column to ``dag_model`` table'
    with op.batch_alter_table('dag') as batch_op:
        batch_op.drop_column('max_active_runs')
    with op.batch_alter_table('dag_run', schema=None) as batch_op:
        batch_op.drop_index('idx_dag_run_dag_id')
        batch_op.drop_index('idx_dag_run_running_dags')