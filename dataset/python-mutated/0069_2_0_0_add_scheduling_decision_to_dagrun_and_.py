"""Add ``scheduling_decision`` to ``DagRun`` and ``DAG``

Revision ID: 98271e7606e2
Revises: bef4f3d11e8b
Create Date: 2020-10-01 12:13:32.968148

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.db_types import TIMESTAMP
revision = '98271e7606e2'
down_revision = 'bef4f3d11e8b'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        while True:
            i = 10
    'Apply Add ``scheduling_decision`` to ``DagRun`` and ``DAG``'
    conn = op.get_bind()
    is_sqlite = bool(conn.dialect.name == 'sqlite')
    is_mssql = bool(conn.dialect.name == 'mssql')
    if is_sqlite:
        op.execute('PRAGMA foreign_keys=off')
    with op.batch_alter_table('dag_run', schema=None) as batch_op:
        batch_op.add_column(sa.Column('last_scheduling_decision', TIMESTAMP, nullable=True))
        batch_op.create_index('idx_last_scheduling_decision', ['last_scheduling_decision'], unique=False)
        batch_op.add_column(sa.Column('dag_hash', sa.String(32), nullable=True))
    with op.batch_alter_table('dag', schema=None) as batch_op:
        batch_op.add_column(sa.Column('next_dagrun', TIMESTAMP, nullable=True))
        batch_op.add_column(sa.Column('next_dagrun_create_after', TIMESTAMP, nullable=True))
        batch_op.add_column(sa.Column('concurrency', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('has_task_concurrency_limits', sa.Boolean(), nullable=True))
        batch_op.create_index('idx_next_dagrun_create_after', ['next_dagrun_create_after'], unique=False)
    try:
        from airflow.configuration import conf
        concurrency = conf.getint('core', 'max_active_tasks_per_dag', fallback=16)
    except:
        concurrency = 16
    op.execute(f'\n        UPDATE dag SET\n            concurrency={concurrency},\n            has_task_concurrency_limits={(1 if is_sqlite or is_mssql else sa.true())}\n        where concurrency IS NULL\n        ')
    with op.batch_alter_table('dag', schema=None) as batch_op:
        batch_op.alter_column('concurrency', type_=sa.Integer(), nullable=False)
        batch_op.alter_column('has_task_concurrency_limits', type_=sa.Boolean(), nullable=False)
    if is_sqlite:
        op.execute('PRAGMA foreign_keys=on')

def downgrade():
    if False:
        return 10
    'Unapply Add ``scheduling_decision`` to ``DagRun`` and ``DAG``'
    conn = op.get_bind()
    is_sqlite = bool(conn.dialect.name == 'sqlite')
    if is_sqlite:
        op.execute('PRAGMA foreign_keys=off')
    with op.batch_alter_table('dag_run', schema=None) as batch_op:
        batch_op.drop_index('idx_last_scheduling_decision')
        batch_op.drop_column('last_scheduling_decision')
        batch_op.drop_column('dag_hash')
    with op.batch_alter_table('dag', schema=None) as batch_op:
        batch_op.drop_index('idx_next_dagrun_create_after')
        batch_op.drop_column('next_dagrun_create_after')
        batch_op.drop_column('next_dagrun')
        batch_op.drop_column('concurrency')
        batch_op.drop_column('has_task_concurrency_limits')
    if is_sqlite:
        op.execute('PRAGMA foreign_keys=on')