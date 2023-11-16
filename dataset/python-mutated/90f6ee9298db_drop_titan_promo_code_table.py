"""
Drop titan_promo_code table

Revision ID: 90f6ee9298db
Revises: d0f67adbcb80
Create Date: 2022-10-03 18:48:39.327937
"""
from alembic import op
revision = '90f6ee9298db'
down_revision = 'd0f67adbcb80'

def upgrade():
    if False:
        return 10
    op.drop_index('ix_user_titan_codes_user_id', table_name='user_titan_codes')
    op.drop_table('user_titan_codes')

def downgrade():
    if False:
        i = 10
        return i + 15
    raise RuntimeError("Can't roll back")