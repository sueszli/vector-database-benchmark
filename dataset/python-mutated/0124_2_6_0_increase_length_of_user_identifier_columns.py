"""Increase length of user identifier columns in ``ab_user`` and ``ab_register_user`` tables

Revision ID: 98ae134e6fff
Revises: 6abdffdd4815
Create Date: 2023-01-18 16:21:09.420958

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.utils import get_mssql_table_constraints
revision = '98ae134e6fff'
down_revision = '6abdffdd4815'
branch_labels = None
depends_on = None
airflow_version = '2.6.0'

def upgrade():
    if False:
        while True:
            i = 10
    'Increase length of user identifier columns in ab_user and ab_register_user tables'
    with op.batch_alter_table('ab_user') as batch_op:
        batch_op.alter_column('first_name', type_=sa.String(256), existing_nullable=False)
        batch_op.alter_column('last_name', type_=sa.String(256), existing_nullable=False)
        batch_op.alter_column('username', type_=sa.String(512).with_variant(sa.String(512, collation='NOCASE'), 'sqlite'), existing_nullable=False)
        batch_op.alter_column('email', type_=sa.String(512), existing_nullable=False)
    with op.batch_alter_table('ab_register_user') as batch_op:
        batch_op.alter_column('first_name', type_=sa.String(256), existing_nullable=False)
        batch_op.alter_column('last_name', type_=sa.String(256), existing_nullable=False)
        batch_op.alter_column('username', type_=sa.String(512).with_variant(sa.String(512, collation='NOCASE'), 'sqlite'), existing_nullable=False)
        batch_op.alter_column('email', type_=sa.String(512), existing_nullable=False)

def downgrade():
    if False:
        i = 10
        return i + 15
    'Revert length of user identifier columns in ab_user and ab_register_user tables'
    conn = op.get_bind()
    if conn.dialect.name != 'mssql':
        with op.batch_alter_table('ab_user') as batch_op:
            batch_op.alter_column('first_name', type_=sa.String(64), existing_nullable=False)
            batch_op.alter_column('last_name', type_=sa.String(64), existing_nullable=False)
            batch_op.alter_column('username', type_=sa.String(256).with_variant(sa.String(256, collation='NOCASE'), 'sqlite'), existing_nullable=False)
            batch_op.alter_column('email', type_=sa.String(256), existing_nullable=False)
        with op.batch_alter_table('ab_register_user') as batch_op:
            batch_op.alter_column('first_name', type_=sa.String(64), existing_nullable=False)
            batch_op.alter_column('last_name', type_=sa.String(64), existing_nullable=False)
            batch_op.alter_column('username', type_=sa.String(256).with_variant(sa.String(256, collation='NOCASE'), 'sqlite'), existing_nullable=False)
            batch_op.alter_column('email', type_=sa.String(256), existing_nullable=False)
    else:
        with op.batch_alter_table('ab_user') as batch_op:
            batch_op.alter_column('first_name', type_=sa.String(64), existing_nullable=False)
            batch_op.alter_column('last_name', type_=sa.String(64), existing_nullable=False)
            constraints = get_mssql_table_constraints(conn, 'ab_user')
            for (k, _) in constraints.get('UNIQUE').items():
                batch_op.drop_constraint(k, type_='unique')
            batch_op.alter_column('username', type_=sa.String(256), existing_nullable=False)
            batch_op.create_unique_constraint(None, ['username'])
            batch_op.alter_column('email', type_=sa.String(256), existing_nullable=False)
            batch_op.create_unique_constraint(None, ['email'])
        with op.batch_alter_table('ab_register_user') as batch_op:
            batch_op.alter_column('first_name', type_=sa.String(64), existing_nullable=False)
            batch_op.alter_column('last_name', type_=sa.String(64), existing_nullable=False)
            batch_op.alter_column('email', type_=sa.String(256), existing_nullable=False)
            constraints = get_mssql_table_constraints(conn, 'ab_register_user')
            for (k, _) in constraints.get('UNIQUE').items():
                batch_op.drop_constraint(k, type_='unique')
            batch_op.alter_column('username', type_=sa.String(256), existing_nullable=False)
            batch_op.create_unique_constraint(None, ['username'])