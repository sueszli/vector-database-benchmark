"""Increase length of ``Flask-AppBuilder`` ``ab_view_menu.name`` column

Revision ID: 03afc6b6f902
Revises: 92c57b58940d
Create Date: 2020-11-13 22:21:41.619565

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect
from airflow.migrations.db_types import StringID
revision = '03afc6b6f902'
down_revision = '92c57b58940d'
branch_labels = None
depends_on = None
airflow_version = '1.10.13'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Apply Increase length of ``Flask-AppBuilder`` ``ab_view_menu.name`` column'
    conn = op.get_bind()
    inspector = inspect(conn)
    tables = inspector.get_table_names()
    if 'ab_view_menu' in tables:
        if conn.dialect.name == 'sqlite':
            op.execute('PRAGMA foreign_keys=off')
            op.execute('\n            CREATE TABLE IF NOT EXISTS ab_view_menu_dg_tmp\n            (\n                id INTEGER NOT NULL PRIMARY KEY,\n                name VARCHAR(250) NOT NULL UNIQUE\n            );\n            ')
            op.execute('INSERT INTO ab_view_menu_dg_tmp(id, name) select id, name from ab_view_menu;')
            op.execute('DROP TABLE ab_view_menu')
            op.execute('ALTER TABLE ab_view_menu_dg_tmp rename to ab_view_menu;')
            op.execute('PRAGMA foreign_keys=on')
        else:
            op.alter_column(table_name='ab_view_menu', column_name='name', type_=StringID(length=250), nullable=False)

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Increase length of ``Flask-AppBuilder`` ``ab_view_menu.name`` column'
    conn = op.get_bind()
    inspector = inspect(conn)
    tables = inspector.get_table_names()
    if 'ab_view_menu' in tables:
        if conn.dialect.name == 'sqlite':
            op.execute('PRAGMA foreign_keys=off')
            op.execute('\n                CREATE TABLE IF NOT EXISTS ab_view_menu_dg_tmp\n                (\n                    id INTEGER NOT NULL PRIMARY KEY,\n                    name VARCHAR(100) NOT NULL UNIQUE\n                );\n                ')
            op.execute('INSERT INTO ab_view_menu_dg_tmp(id, name) select id, name from ab_view_menu;')
            op.execute('DROP TABLE ab_view_menu')
            op.execute('ALTER TABLE ab_view_menu_dg_tmp rename to ab_view_menu;')
            op.execute('PRAGMA foreign_keys=on')
        else:
            op.alter_column(table_name='ab_view_menu', column_name='name', type_=sa.String(length=100), nullable=False)