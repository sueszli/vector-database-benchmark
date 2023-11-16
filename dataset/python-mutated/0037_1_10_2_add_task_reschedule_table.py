"""Add ``task_reschedule`` table

Revision ID: 0a2a5b66e19d
Revises: 9635ae0956e7
Create Date: 2018-06-17 22:50:00.053620

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.db_types import TIMESTAMP, StringID
revision = '0a2a5b66e19d'
down_revision = '9635ae0956e7'
branch_labels = None
depends_on = None
airflow_version = '1.10.2'
TABLE_NAME = 'task_reschedule'
INDEX_NAME = 'idx_' + TABLE_NAME + '_dag_task_date'

def upgrade():
    if False:
        while True:
            i = 10
    timestamp = TIMESTAMP
    if op.get_bind().dialect.name == 'mssql':
        timestamp = sa.DateTime()
    op.create_table(TABLE_NAME, sa.Column('id', sa.Integer(), nullable=False), sa.Column('task_id', StringID(), nullable=False), sa.Column('dag_id', StringID(), nullable=False), sa.Column('execution_date', timestamp, nullable=False, server_default=None), sa.Column('try_number', sa.Integer(), nullable=False), sa.Column('start_date', timestamp, nullable=False), sa.Column('end_date', timestamp, nullable=False), sa.Column('duration', sa.Integer(), nullable=False), sa.Column('reschedule_date', timestamp, nullable=False), sa.PrimaryKeyConstraint('id'), sa.ForeignKeyConstraint(['task_id', 'dag_id', 'execution_date'], ['task_instance.task_id', 'task_instance.dag_id', 'task_instance.execution_date'], name='task_reschedule_dag_task_date_fkey'))
    op.create_index(INDEX_NAME, TABLE_NAME, ['dag_id', 'task_id', 'execution_date'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index(INDEX_NAME, table_name=TABLE_NAME)
    op.drop_table(TABLE_NAME)