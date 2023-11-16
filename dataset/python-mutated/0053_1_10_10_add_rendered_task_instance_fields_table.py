"""Add ``RenderedTaskInstanceFields`` table

Revision ID: 852ae6c715af
Revises: a4c2fd67d16b
Create Date: 2020-03-10 22:19:18.034961

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy import text
from airflow.migrations.db_types import StringID
revision = '852ae6c715af'
down_revision = 'a4c2fd67d16b'
branch_labels = None
depends_on = None
airflow_version = '1.10.10'
TABLE_NAME = 'rendered_task_instance_fields'

def upgrade():
    if False:
        while True:
            i = 10
    'Apply Add ``RenderedTaskInstanceFields`` table'
    json_type = sa.JSON
    conn = op.get_bind()
    if conn.dialect.name != 'postgresql':
        try:
            conn.execute(text('SELECT JSON_VALID(1)')).fetchone()
        except (sa.exc.OperationalError, sa.exc.ProgrammingError):
            json_type = sa.Text
    op.create_table(TABLE_NAME, sa.Column('dag_id', StringID(), nullable=False), sa.Column('task_id', StringID(), nullable=False), sa.Column('execution_date', sa.TIMESTAMP(timezone=True), nullable=False), sa.Column('rendered_fields', json_type(), nullable=False), sa.PrimaryKeyConstraint('dag_id', 'task_id', 'execution_date'))

def downgrade():
    if False:
        i = 10
        return i + 15
    'Drop RenderedTaskInstanceFields table'
    op.drop_table(TABLE_NAME)