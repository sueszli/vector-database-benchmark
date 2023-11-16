"""Increase maximum length of pool name in ``task_instance`` table to ``256`` characters

Revision ID: 90d1635d7b86
Revises: 2e42bb497a22
Create Date: 2021-04-05 09:37:54.848731

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '90d1635d7b86'
down_revision = '2e42bb497a22'
branch_labels = None
depends_on = None
airflow_version = '2.1.0'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Apply Increase maximum length of pool name in ``task_instance`` table to ``256`` characters'
    with op.batch_alter_table('task_instance') as batch_op:
        batch_op.alter_column('pool', type_=sa.String(256), nullable=False)

def downgrade():
    if False:
        while True:
            i = 10
    'Unapply Increase maximum length of pool name in ``task_instance`` table to ``256`` characters'
    conn = op.get_bind()
    if conn.dialect.name == 'mssql':
        with op.batch_alter_table('task_instance') as batch_op:
            batch_op.drop_index('ti_pool')
            batch_op.alter_column('pool', type_=sa.String(50), nullable=False)
            batch_op.create_index('ti_pool', ['pool'])
    else:
        with op.batch_alter_table('task_instance') as batch_op:
            batch_op.alter_column('pool', type_=sa.String(50), nullable=False)