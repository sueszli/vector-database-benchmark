"""Add ``sensor_instance`` table

Revision ID: e38be357a868
Revises: 8d48763f6d53
Create Date: 2019-06-07 04:03:17.003939

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy import func, inspect
from airflow.migrations.db_types import TIMESTAMP, StringID
revision = 'e38be357a868'
down_revision = '8d48763f6d53'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        while True:
            i = 10
    conn = op.get_bind()
    inspector = inspect(conn)
    tables = inspector.get_table_names()
    if 'sensor_instance' in tables:
        return
    op.create_table('sensor_instance', sa.Column('id', sa.Integer(), nullable=False), sa.Column('task_id', StringID(), nullable=False), sa.Column('dag_id', StringID(), nullable=False), sa.Column('execution_date', TIMESTAMP, nullable=False), sa.Column('state', sa.String(length=20), nullable=True), sa.Column('try_number', sa.Integer(), nullable=True), sa.Column('start_date', TIMESTAMP, nullable=True), sa.Column('operator', sa.String(length=1000), nullable=False), sa.Column('op_classpath', sa.String(length=1000), nullable=False), sa.Column('hashcode', sa.BigInteger(), nullable=False), sa.Column('shardcode', sa.Integer(), nullable=False), sa.Column('poke_context', sa.Text(), nullable=False), sa.Column('execution_context', sa.Text(), nullable=True), sa.Column('created_at', TIMESTAMP, default=func.now, nullable=False), sa.Column('updated_at', TIMESTAMP, default=func.now, nullable=False), sa.PrimaryKeyConstraint('id'))
    op.create_index('ti_primary_key', 'sensor_instance', ['dag_id', 'task_id', 'execution_date'], unique=True)
    op.create_index('si_hashcode', 'sensor_instance', ['hashcode'], unique=False)
    op.create_index('si_shardcode', 'sensor_instance', ['shardcode'], unique=False)
    op.create_index('si_state_shard', 'sensor_instance', ['state', 'shardcode'], unique=False)
    op.create_index('si_updated_at', 'sensor_instance', ['updated_at'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    conn = op.get_bind()
    inspector = inspect(conn)
    tables = inspector.get_table_names()
    if 'sensor_instance' in tables:
        op.drop_table('sensor_instance')