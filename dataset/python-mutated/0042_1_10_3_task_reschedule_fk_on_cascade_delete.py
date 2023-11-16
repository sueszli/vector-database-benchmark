"""task reschedule foreign key on cascade delete

Revision ID: 939bb1e647c8
Revises: dd4ecb8fbee3
Create Date: 2019-02-04 20:21:50.669751

"""
from __future__ import annotations
from alembic import op
revision = '939bb1e647c8'
down_revision = 'dd4ecb8fbee3'
branch_labels = None
depends_on = None
airflow_version = '1.10.3'

def upgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('task_reschedule') as batch_op:
        batch_op.drop_constraint('task_reschedule_dag_task_date_fkey', type_='foreignkey')
        batch_op.create_foreign_key('task_reschedule_dag_task_date_fkey', 'task_instance', ['task_id', 'dag_id', 'execution_date'], ['task_id', 'dag_id', 'execution_date'], ondelete='CASCADE')

def downgrade():
    if False:
        return 10
    with op.batch_alter_table('task_reschedule') as batch_op:
        batch_op.drop_constraint('task_reschedule_dag_task_date_fkey', type_='foreignkey')
        batch_op.create_foreign_key('task_reschedule_dag_task_date_fkey', 'task_instance', ['task_id', 'dag_id', 'execution_date'], ['task_id', 'dag_id', 'execution_date'])