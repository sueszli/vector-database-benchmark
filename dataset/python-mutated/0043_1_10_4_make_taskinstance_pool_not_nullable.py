"""Make ``TaskInstance.pool`` not nullable

Revision ID: 6e96a59344a4
Revises: 939bb1e647c8
Create Date: 2019-06-13 21:51:32.878437

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy import Column, String
from sqlalchemy.orm import declarative_base
from airflow.utils.session import create_session
from airflow.utils.sqlalchemy import UtcDateTime
revision = '6e96a59344a4'
down_revision = '939bb1e647c8'
branch_labels = None
depends_on = None
airflow_version = '1.10.4'
Base = declarative_base()
ID_LEN = 250

class TaskInstance(Base):
    """Minimal model definition for migrations"""
    __tablename__ = 'task_instance'
    task_id = Column(String(), primary_key=True)
    dag_id = Column(String(), primary_key=True)
    execution_date = Column(UtcDateTime, primary_key=True)
    pool = Column(String(50), nullable=False)

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Make TaskInstance.pool field not nullable.'
    with create_session() as session:
        session.query(TaskInstance).filter(TaskInstance.pool.is_(None)).update({TaskInstance.pool: 'default_pool'}, synchronize_session=False)
        session.commit()
    conn = op.get_bind()
    if conn.dialect.name == 'mssql':
        op.drop_index(index_name='ti_pool', table_name='task_instance')
    with op.batch_alter_table('task_instance') as batch_op:
        batch_op.alter_column(column_name='pool', type_=sa.String(50), nullable=False)
    if conn.dialect.name == 'mssql':
        op.create_index(index_name='ti_pool', table_name='task_instance', columns=['pool', 'state', 'priority_weight'])

def downgrade():
    if False:
        i = 10
        return i + 15
    'Make TaskInstance.pool field nullable.'
    conn = op.get_bind()
    if conn.dialect.name == 'mssql':
        op.drop_index(index_name='ti_pool', table_name='task_instance')
    with op.batch_alter_table('task_instance') as batch_op:
        batch_op.alter_column(column_name='pool', type_=sa.String(50), nullable=True)
    if conn.dialect.name == 'mssql':
        op.create_index(index_name='ti_pool', table_name='task_instance', columns=['pool', 'state', 'priority_weight'])
    with create_session() as session:
        session.query(TaskInstance).filter(TaskInstance.pool == 'default_pool').update({TaskInstance.pool: None}, synchronize_session=False)
        session.commit()