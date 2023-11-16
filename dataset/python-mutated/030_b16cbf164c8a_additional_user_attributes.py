"""030 Additional user_attributes

Revision ID: b16cbf164c8a
Revises: 1bfdf4240915
Create Date: 2018-09-04 18:48:59.340276

"""
from alembic import op
import sqlalchemy as sa
from ckan.migration import skip_based_on_legacy_engine_version
revision = 'b16cbf164c8a'
down_revision = '1bfdf4240915'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    op.add_column('user', sa.Column('openid', sa.UnicodeText))
    op.add_column('user', sa.Column('password', sa.UnicodeText))
    op.add_column('user', sa.Column('fullname', sa.UnicodeText))
    op.add_column('user', sa.Column('email', sa.UnicodeText))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('user', 'openid')
    op.drop_column('user', 'password')
    op.drop_column('user', 'fullname')
    op.drop_column('user', 'email')