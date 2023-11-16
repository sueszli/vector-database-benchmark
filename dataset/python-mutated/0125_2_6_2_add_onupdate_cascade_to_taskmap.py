"""Add ``onupdate`` cascade to ``task_map`` table

Revision ID: c804e5c76e3e
Revises: 98ae134e6fff
Create Date: 2023-05-19 23:30:57.368617

"""
from __future__ import annotations
from alembic import op
revision = 'c804e5c76e3e'
down_revision = '98ae134e6fff'
branch_labels = None
depends_on = None
airflow_version = '2.6.2'

def upgrade():
    if False:
        while True:
            i = 10
    'Apply Add onupdate cascade to taskmap'
    with op.batch_alter_table('task_map') as batch_op:
        batch_op.drop_constraint('task_map_task_instance_fkey', type_='foreignkey')
        batch_op.create_foreign_key('task_map_task_instance_fkey', 'task_instance', ['dag_id', 'task_id', 'run_id', 'map_index'], ['dag_id', 'task_id', 'run_id', 'map_index'], ondelete='CASCADE', onupdate='CASCADE')

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Add onupdate cascade to taskmap'
    with op.batch_alter_table('task_map') as batch_op:
        batch_op.drop_constraint('task_map_task_instance_fkey', type_='foreignkey')
        batch_op.create_foreign_key('task_map_task_instance_fkey', 'task_instance', ['dag_id', 'task_id', 'run_id', 'map_index'], ['dag_id', 'task_id', 'run_id', 'map_index'], ondelete='CASCADE')