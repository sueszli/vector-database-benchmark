"""Add ``task_fail`` table

Revision ID: 64de9cddf6c9
Revises: 211e584da130
Create Date: 2016-08-03 14:02:59.203021

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.db_types import StringID
revision = '64de9cddf6c9'
down_revision = '211e584da130'
branch_labels = None
depends_on = None
airflow_version = '1.7.1.3'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('task_fail', sa.Column('id', sa.Integer(), nullable=False), sa.Column('task_id', StringID(), nullable=False), sa.Column('dag_id', StringID(), nullable=False), sa.Column('execution_date', sa.DateTime(), nullable=False), sa.Column('start_date', sa.DateTime(), nullable=True), sa.Column('end_date', sa.DateTime(), nullable=True), sa.Column('duration', sa.Integer(), nullable=True), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_table('task_fail')