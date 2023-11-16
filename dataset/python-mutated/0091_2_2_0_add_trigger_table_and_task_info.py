"""Adds ``trigger`` table and deferrable operator columns to task instance

Revision ID: 54bebd308c5f
Revises: 30867afad44a
Create Date: 2021-04-14 12:56:40.688260

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.utils.sqlalchemy import ExtendedJSON
revision = '54bebd308c5f'
down_revision = '30867afad44a'
branch_labels = None
depends_on = None
airflow_version = '2.2.0'

def upgrade():
    if False:
        while True:
            i = 10
    'Apply Adds ``trigger`` table and deferrable operator columns to task instance'
    op.create_table('trigger', sa.Column('id', sa.Integer(), primary_key=True, nullable=False), sa.Column('classpath', sa.String(length=1000), nullable=False), sa.Column('kwargs', ExtendedJSON(), nullable=False), sa.Column('created_date', sa.DateTime(), nullable=False), sa.Column('triggerer_id', sa.Integer(), nullable=True))
    with op.batch_alter_table('task_instance', schema=None) as batch_op:
        batch_op.add_column(sa.Column('trigger_id', sa.Integer()))
        batch_op.add_column(sa.Column('trigger_timeout', sa.DateTime()))
        batch_op.add_column(sa.Column('next_method', sa.String(length=1000)))
        batch_op.add_column(sa.Column('next_kwargs', ExtendedJSON()))
        batch_op.create_foreign_key('task_instance_trigger_id_fkey', 'trigger', ['trigger_id'], ['id'], ondelete='CASCADE')
        batch_op.create_index('ti_trigger_id', ['trigger_id'])

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Unapply Adds ``trigger`` table and deferrable operator columns to task instance'
    with op.batch_alter_table('task_instance', schema=None) as batch_op:
        batch_op.drop_constraint('task_instance_trigger_id_fkey', type_='foreignkey')
        batch_op.drop_index('ti_trigger_id')
        batch_op.drop_column('trigger_id')
        batch_op.drop_column('trigger_timeout')
        batch_op.drop_column('next_method')
        batch_op.drop_column('next_kwargs')
    op.drop_table('trigger')