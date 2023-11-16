"""Add index on state, dag_id for queued ``dagrun``

Revision ID: ccde3e26fe78
Revises: 092435bf5d12
Create Date: 2021-09-08 16:35:34.867711

"""
from __future__ import annotations
from alembic import op
from sqlalchemy import text
revision = 'ccde3e26fe78'
down_revision = '092435bf5d12'
branch_labels = None
depends_on = None
airflow_version = '2.1.4'

def upgrade():
    if False:
        print('Hello World!')
    'Apply Add index on state, dag_id for queued ``dagrun``'
    with op.batch_alter_table('dag_run') as batch_op:
        batch_op.create_index('idx_dag_run_queued_dags', ['state', 'dag_id'], postgresql_where=text("state='queued'"), mssql_where=text("state='queued'"), sqlite_where=text("state='queued'"))

def downgrade():
    if False:
        i = 10
        return i + 15
    'Unapply Add index on state, dag_id for queued ``dagrun``'
    with op.batch_alter_table('dag_run') as batch_op:
        batch_op.drop_index('idx_dag_run_queued_dags')