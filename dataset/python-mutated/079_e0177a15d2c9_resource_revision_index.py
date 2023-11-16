"""079 Resource revision index

Revision ID: e0177a15d2c9
Revises: ae821876532a
Create Date: 2018-09-04 18:49:16.198887

"""
from alembic import op
from ckan.migration import skip_based_on_legacy_engine_version
revision = 'e0177a15d2c9'
down_revision = 'ae821876532a'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    op.create_index('idx_resource_continuity_id', 'resource_revision', ['continuity_id'])

def downgrade():
    if False:
        print('Hello World!')
    op.drop_index('idx_resource_continuity_id', 'resource_revision')