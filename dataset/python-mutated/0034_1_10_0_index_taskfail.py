"""Create index on ``task_fail`` table

Revision ID: 9635ae0956e7
Revises: 856955da8476
Create Date: 2018-06-17 21:40:01.963540

"""
from __future__ import annotations
from alembic import op
revision = '9635ae0956e7'
down_revision = '856955da8476'
branch_labels = None
depends_on = None
airflow_version = '1.10.0'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_index('idx_task_fail_dag_task_date', 'task_fail', ['dag_id', 'task_id', 'execution_date'], unique=False)

def downgrade():
    if False:
        return 10
    op.drop_index('idx_task_fail_dag_task_date', table_name='task_fail')