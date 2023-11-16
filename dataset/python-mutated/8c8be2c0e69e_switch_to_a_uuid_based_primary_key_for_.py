"""
Switch to a UUID based primary key for User

Revision ID: 8c8be2c0e69e
Revises: 039f45e2dbf9
Create Date: 2016-07-01 18:20:42.072664
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '8c8be2c0e69e'
down_revision = '039f45e2dbf9'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('accounts_user', sa.Column('new_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False))
    op.add_column('accounts_email', sa.Column('new_user_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('accounts_gpgkey', sa.Column('new_user_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.execute(' UPDATE accounts_email\n            SET new_user_id = accounts_user.new_id\n            FROM accounts_user\n            WHERE accounts_email.user_id = accounts_user.id\n        ')
    op.execute(' UPDATE accounts_gpgkey\n            SET new_user_id = accounts_user.new_id\n            FROM accounts_user\n            WHERE accounts_gpgkey.user_id = accounts_user.id\n        ')
    op.alter_column('accounts_email', 'new_user_id', nullable=False)
    op.alter_column('accounts_gpgkey', 'new_user_id', nullable=False)
    op.drop_constraint('accounts_email_user_id_fkey', 'accounts_email')
    op.drop_column('accounts_email', 'user_id')
    op.alter_column('accounts_email', 'new_user_id', new_column_name='user_id')
    op.drop_constraint('accounts_gpgkey_user_id_fkey', 'accounts_gpgkey')
    op.drop_column('accounts_gpgkey', 'user_id')
    op.alter_column('accounts_gpgkey', 'new_user_id', new_column_name='user_id')
    op.drop_constraint('accounts_user_pkey', 'accounts_user')
    op.create_primary_key(None, 'accounts_user', ['new_id'])
    op.drop_column('accounts_user', 'id')
    op.alter_column('accounts_user', 'new_id', new_column_name='id')
    op.create_foreign_key(None, 'accounts_email', 'accounts_user', ['user_id'], ['id'], deferrable=True)
    op.create_foreign_key(None, 'accounts_gpgkey', 'accounts_user', ['user_id'], ['id'], deferrable=True)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    raise RuntimeError('Order No. 227 - Ни шагу назад!')