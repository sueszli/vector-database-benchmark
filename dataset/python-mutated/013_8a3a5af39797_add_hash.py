"""013 Add hash

Revision ID: 8a3a5af39797
Revises: e5ca33a5d445
Create Date: 2018-09-04 18:48:52.640250

"""
from alembic import op
import sqlalchemy as sa
from ckan.migration import skip_based_on_legacy_engine_version
revision = '8a3a5af39797'
down_revision = 'e5ca33a5d445'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    op.add_column('package_resource', sa.Column('hash', sa.UnicodeText))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('package_resource', 'hash')