"""Fix MySQL not null constraint

Revision ID: f23433877c24
Revises: 05f30312d566
Create Date: 2018-06-17 10:16:31.412131

"""
from __future__ import annotations
from alembic import op
from sqlalchemy import text
from sqlalchemy.dialects import mysql
revision = 'f23433877c24'
down_revision = '05f30312d566'
branch_labels = None
depends_on = None
airflow_version = '1.10.0'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
        conn.execute(text("SET time_zone = '+00:00'"))
        op.alter_column('task_fail', 'execution_date', existing_type=mysql.TIMESTAMP(fsp=6), nullable=False)
        op.alter_column('xcom', 'execution_date', existing_type=mysql.TIMESTAMP(fsp=6), nullable=False)
        op.alter_column('xcom', 'timestamp', existing_type=mysql.TIMESTAMP(fsp=6), nullable=False)

def downgrade():
    if False:
        print('Hello World!')
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
        conn.execute(text("SET time_zone = '+00:00'"))
        op.alter_column('xcom', 'timestamp', existing_type=mysql.TIMESTAMP(fsp=6), nullable=True)
        op.alter_column('xcom', 'execution_date', existing_type=mysql.TIMESTAMP(fsp=6), nullable=True)
        op.alter_column('task_fail', 'execution_date', existing_type=mysql.TIMESTAMP(fsp=6), nullable=True)