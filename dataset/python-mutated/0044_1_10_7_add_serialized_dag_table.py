"""Add ``serialized_dag`` table

Revision ID: d38e04c12aa2
Revises: 6e96a59344a4
Create Date: 2019-08-01 14:39:35.616417

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy import text
from sqlalchemy.dialects import mysql
from airflow.migrations.db_types import StringID
revision = 'd38e04c12aa2'
down_revision = '6e96a59344a4'
branch_labels = None
depends_on = None
airflow_version = '1.10.7'

def upgrade():
    if False:
        while True:
            i = 10
    'Upgrade version.'
    json_type = sa.JSON
    conn = op.get_bind()
    if conn.dialect.name != 'postgresql':
        try:
            conn.execute(text('SELECT JSON_VALID(1)')).fetchone()
        except (sa.exc.OperationalError, sa.exc.ProgrammingError):
            json_type = sa.Text
    op.create_table('serialized_dag', sa.Column('dag_id', StringID(), nullable=False), sa.Column('fileloc', sa.String(length=2000), nullable=False), sa.Column('fileloc_hash', sa.Integer(), nullable=False), sa.Column('data', json_type(), nullable=False), sa.Column('last_updated', sa.DateTime(), nullable=False), sa.PrimaryKeyConstraint('dag_id'))
    op.create_index('idx_fileloc_hash', 'serialized_dag', ['fileloc_hash'])
    if conn.dialect.name == 'mysql':
        conn.execute(text("SET time_zone = '+00:00'"))
        cur = conn.execute(text('SELECT @@explicit_defaults_for_timestamp'))
        res = cur.fetchall()
        if res[0][0] == 0:
            raise Exception('Global variable explicit_defaults_for_timestamp needs to be on (1) for mysql')
        op.alter_column(table_name='serialized_dag', column_name='last_updated', type_=mysql.TIMESTAMP(fsp=6), nullable=False)
    else:
        if conn.dialect.name in ('sqlite', 'mssql'):
            return
        if conn.dialect.name == 'postgresql':
            conn.execute(text('set timezone=UTC'))
        op.alter_column(table_name='serialized_dag', column_name='last_updated', type_=sa.TIMESTAMP(timezone=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Downgrade version.'
    op.drop_table('serialized_dag')