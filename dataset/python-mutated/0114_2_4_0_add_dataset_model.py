"""Add Dataset model

Revision ID: 0038cd0c28b4
Revises: 44b7034f6bdc
Create Date: 2022-06-22 14:37:20.880672

"""
from __future__ import annotations
import sqlalchemy as sa
import sqlalchemy_jsonfield
from alembic import op
from sqlalchemy import Integer, String, func
from airflow.migrations.db_types import TIMESTAMP, StringID
from airflow.settings import json
revision = '0038cd0c28b4'
down_revision = '44b7034f6bdc'
branch_labels = None
depends_on = None
airflow_version = '2.4.0'

def _create_dataset_table():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('dataset', sa.Column('id', Integer, primary_key=True, autoincrement=True), sa.Column('uri', String(length=3000).with_variant(String(length=3000, collation='latin1_general_cs'), 'mysql'), nullable=False), sa.Column('extra', sqlalchemy_jsonfield.JSONField(json=json), nullable=False, default={}), sa.Column('created_at', TIMESTAMP, nullable=False), sa.Column('updated_at', TIMESTAMP, nullable=False), sqlite_autoincrement=True)
    op.create_index('idx_uri_unique', 'dataset', ['uri'], unique=True)

def _create_dag_schedule_dataset_reference_table():
    if False:
        i = 10
        return i + 15
    op.create_table('dag_schedule_dataset_reference', sa.Column('dataset_id', Integer, primary_key=True, nullable=False), sa.Column('dag_id', StringID(), primary_key=True, nullable=False), sa.Column('created_at', TIMESTAMP, default=func.now, nullable=False), sa.Column('updated_at', TIMESTAMP, default=func.now, nullable=False), sa.ForeignKeyConstraint(('dataset_id',), ['dataset.id'], name='dsdr_dataset_fkey', ondelete='CASCADE'), sa.ForeignKeyConstraint(columns=('dag_id',), refcolumns=['dag.dag_id'], name='dsdr_dag_id_fkey', ondelete='CASCADE'))

def _create_task_outlet_dataset_reference_table():
    if False:
        i = 10
        return i + 15
    op.create_table('task_outlet_dataset_reference', sa.Column('dataset_id', Integer, primary_key=True, nullable=False), sa.Column('dag_id', StringID(), primary_key=True, nullable=False), sa.Column('task_id', StringID(), primary_key=True, nullable=False), sa.Column('created_at', TIMESTAMP, default=func.now, nullable=False), sa.Column('updated_at', TIMESTAMP, default=func.now, nullable=False), sa.ForeignKeyConstraint(('dataset_id',), ['dataset.id'], name='todr_dataset_fkey', ondelete='CASCADE'), sa.ForeignKeyConstraint(columns=('dag_id',), refcolumns=['dag.dag_id'], name='todr_dag_id_fkey', ondelete='CASCADE'))

def _create_dataset_dag_run_queue_table():
    if False:
        return 10
    op.create_table('dataset_dag_run_queue', sa.Column('dataset_id', Integer, primary_key=True, nullable=False), sa.Column('target_dag_id', StringID(), primary_key=True, nullable=False), sa.Column('created_at', TIMESTAMP, default=func.now, nullable=False), sa.ForeignKeyConstraint(('dataset_id',), ['dataset.id'], name='ddrq_dataset_fkey', ondelete='CASCADE'), sa.ForeignKeyConstraint(('target_dag_id',), ['dag.dag_id'], name='ddrq_dag_fkey', ondelete='CASCADE'))

def _create_dataset_event_table():
    if False:
        print('Hello World!')
    op.create_table('dataset_event', sa.Column('id', Integer, primary_key=True, autoincrement=True), sa.Column('dataset_id', Integer, nullable=False), sa.Column('extra', sqlalchemy_jsonfield.JSONField(json=json), nullable=False, default={}), sa.Column('source_task_id', String(250), nullable=True), sa.Column('source_dag_id', String(250), nullable=True), sa.Column('source_run_id', String(250), nullable=True), sa.Column('source_map_index', sa.Integer(), nullable=True, server_default='-1'), sa.Column('timestamp', TIMESTAMP, nullable=False), sqlite_autoincrement=True)
    op.create_index('idx_dataset_id_timestamp', 'dataset_event', ['dataset_id', 'timestamp'])

def _create_dataset_event_dag_run_table():
    if False:
        print('Hello World!')
    op.create_table('dagrun_dataset_event', sa.Column('dag_run_id', sa.Integer(), nullable=False), sa.Column('event_id', sa.Integer(), nullable=False), sa.ForeignKeyConstraint(['dag_run_id'], ['dag_run.id'], name=op.f('dagrun_dataset_events_dag_run_id_fkey'), ondelete='CASCADE'), sa.ForeignKeyConstraint(['event_id'], ['dataset_event.id'], name=op.f('dagrun_dataset_events_event_id_fkey'), ondelete='CASCADE'), sa.PrimaryKeyConstraint('dag_run_id', 'event_id', name=op.f('dagrun_dataset_events_pkey')))
    with op.batch_alter_table('dagrun_dataset_event') as batch_op:
        batch_op.create_index('idx_dagrun_dataset_events_dag_run_id', ['dag_run_id'], unique=False)
        batch_op.create_index('idx_dagrun_dataset_events_event_id', ['event_id'], unique=False)

def upgrade():
    if False:
        print('Hello World!')
    'Apply Add Dataset model'
    _create_dataset_table()
    _create_dag_schedule_dataset_reference_table()
    _create_task_outlet_dataset_reference_table()
    _create_dataset_dag_run_queue_table()
    _create_dataset_event_table()
    _create_dataset_event_dag_run_table()

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Unapply Add Dataset model'
    op.drop_table('dag_schedule_dataset_reference')
    op.drop_table('task_outlet_dataset_reference')
    op.drop_table('dataset_dag_run_queue')
    op.drop_table('dagrun_dataset_event')
    op.drop_table('dataset_event')
    op.drop_table('dataset')