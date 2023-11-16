"""Add case-insensitive unique constraint for username

Revision ID: e07f49787c9d
Revises: b0d31815b5a6
Create Date: 2022-10-25 17:29:46.432326

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'e07f49787c9d'
down_revision = 'b0d31815b5a6'
branch_labels = None
depends_on = None
airflow_version = '2.4.3'

def upgrade():
    if False:
        i = 10
        return i + 15
    'Apply Add case-insensitive unique constraint'
    conn = op.get_bind()
    if conn.dialect.name == 'postgresql':
        op.create_index('idx_ab_user_username', 'ab_user', [sa.text('LOWER(username)')], unique=True)
        op.create_index('idx_ab_register_user_username', 'ab_register_user', [sa.text('LOWER(username)')], unique=True)
    elif conn.dialect.name == 'sqlite':
        with op.batch_alter_table('ab_user') as batch_op:
            batch_op.alter_column('username', existing_type=sa.String(64), _type=sa.String(64, collation='NOCASE'), unique=True, nullable=False)
        with op.batch_alter_table('ab_register_user') as batch_op:
            batch_op.alter_column('username', existing_type=sa.String(64), _type=sa.String(64, collation='NOCASE'), unique=True, nullable=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Unapply Add case-insensitive unique constraint'
    conn = op.get_bind()
    if conn.dialect.name == 'postgresql':
        op.drop_index('idx_ab_user_username', table_name='ab_user')
        op.drop_index('idx_ab_register_user_username', table_name='ab_register_user')
    elif conn.dialect.name == 'sqlite':
        with op.batch_alter_table('ab_user') as batch_op:
            batch_op.alter_column('username', existing_type=sa.String(64, collation='NOCASE'), _type=sa.String(64), unique=True, nullable=False)
        with op.batch_alter_table('ab_register_user') as batch_op:
            batch_op.alter_column('username', existing_type=sa.String(64, collation='NOCASE'), _type=sa.String(64), unique=True, nullable=False)