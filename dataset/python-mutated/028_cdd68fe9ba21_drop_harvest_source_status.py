"""028 Drop harvest_source_status

Revision ID: cdd68fe9ba21
Revises: 11e5745c6fc9
Create Date: 2018-09-04 18:48:58.674039

"""
from alembic import op
from ckan.migration import skip_based_on_legacy_engine_version
revision = 'cdd68fe9ba21'
down_revision = '11e5745c6fc9'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    op.alter_column('harvest_source', 'status', nullable=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.alter_column('harvest_source', 'status', nullable=True)