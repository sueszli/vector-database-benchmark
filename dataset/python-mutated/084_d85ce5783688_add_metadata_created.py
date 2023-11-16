"""084 Add metadata created

Revision ID: d85ce5783688
Revises: f98d8fa2a7f7
Create Date: 2018-09-04 18:49:17.957865

"""
from alembic import op
import sqlalchemy as sa
from ckan.migration import skip_based_on_legacy_engine_version
revision = 'd85ce5783688'
down_revision = 'f98d8fa2a7f7'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    op.add_column('package_revision', sa.Column('metadata_created', sa.TIMESTAMP))
    op.add_column('package', sa.Column('metadata_created', sa.TIMESTAMP))
    conn = op.get_bind()
    conn.execute('\n        UPDATE package SET metadata_created=\n            (SELECT revision_timestamp\n             FROM package_revision\n             WHERE id=package.id\n             ORDER BY revision_timestamp ASC\n             LIMIT 1);\n    ')

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('package', 'metadata_created')
    op.drop_column('package_revision', 'metadata_created')