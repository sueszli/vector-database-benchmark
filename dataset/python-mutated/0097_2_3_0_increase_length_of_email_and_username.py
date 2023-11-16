"""Increase length of email and username in ``ab_user`` and ``ab_register_user`` table to ``256`` characters

Revision ID: 5e3ec427fdd3
Revises: 587bdf053233
Create Date: 2021-12-01 11:49:26.390210

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.utils import get_mssql_table_constraints
revision = '5e3ec427fdd3'
down_revision = '587bdf053233'
branch_labels = None
depends_on = None
airflow_version = '2.3.0'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Increase length of email from 64 to 256 characters'
    with op.batch_alter_table('ab_user') as batch_op:
        batch_op.alter_column('username', type_=sa.String(256))
        batch_op.alter_column('email', type_=sa.String(256))
    with op.batch_alter_table('ab_register_user') as batch_op:
        batch_op.alter_column('username', type_=sa.String(256))
        batch_op.alter_column('email', type_=sa.String(256))

def downgrade():
    if False:
        while True:
            i = 10
    'Revert length of email from 256 to 64 characters'
    conn = op.get_bind()
    if conn.dialect.name != 'mssql':
        with op.batch_alter_table('ab_user') as batch_op:
            batch_op.alter_column('username', type_=sa.String(64), nullable=False)
            batch_op.alter_column('email', type_=sa.String(64))
        with op.batch_alter_table('ab_register_user') as batch_op:
            batch_op.alter_column('username', type_=sa.String(64))
            batch_op.alter_column('email', type_=sa.String(64))
    else:
        with op.batch_alter_table('ab_user') as batch_op:
            constraints = get_mssql_table_constraints(conn, 'ab_user')
            (unique_key, _) = constraints['UNIQUE'].popitem()
            batch_op.drop_constraint(unique_key, type_='unique')
            (unique_key, _) = constraints['UNIQUE'].popitem()
            batch_op.drop_constraint(unique_key, type_='unique')
            batch_op.alter_column('username', type_=sa.String(64), nullable=False)
            batch_op.create_unique_constraint(None, ['username'])
            batch_op.alter_column('email', type_=sa.String(64))
            batch_op.create_unique_constraint(None, ['email'])
        with op.batch_alter_table('ab_register_user') as batch_op:
            constraints = get_mssql_table_constraints(conn, 'ab_register_user')
            for (k, _) in constraints.get('UNIQUE').items():
                batch_op.drop_constraint(k, type_='unique')
            batch_op.alter_column('username', type_=sa.String(64))
            batch_op.create_unique_constraint(None, ['username'])
            batch_op.alter_column('email', type_=sa.String(64))