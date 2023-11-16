"""059 Add related count and_flag

Revision ID: 9291bb46f352
Revises: bd36d1826a5d
Create Date: 2018-09-04 18:49:09.243738

"""
from alembic import op
import sqlalchemy as sa
from ckan.migration import skip_based_on_legacy_engine_version
revision = '9291bb46f352'
down_revision = 'bd36d1826a5d'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    op.add_column('related', sa.Column('view_count', sa.Integer, nullable=False, server_default='0'))
    op.add_column('related', sa.Column('featured', sa.Integer, nullable=False, server_default='0'))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('related', 'view_count')
    op.drop_column('related', 'featured')