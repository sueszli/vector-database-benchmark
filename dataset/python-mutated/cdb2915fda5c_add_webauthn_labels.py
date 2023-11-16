"""
add webauthn labels

Revision ID: cdb2915fda5c
Revises: af7dca2bb2fe
Create Date: 2019-06-08 16:31:41.681380
"""
import sqlalchemy as sa
from alembic import op
revision = 'cdb2915fda5c'
down_revision = 'af7dca2bb2fe'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('user_security_keys', sa.Column('label', sa.String(), nullable=False))
    op.create_index('user_security_keys_label_key', 'user_security_keys', ['user_id'], unique=False)

def downgrade():
    if False:
        return 10
    op.drop_index('user_security_keys_label_key', table_name='user_security_keys')
    op.drop_column('user_security_keys', 'label')