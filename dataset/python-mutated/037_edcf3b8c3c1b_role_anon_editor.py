"""037 Role anon_editor

Revision ID: edcf3b8c3c1b
Revises: ecaa8b38782f
Create Date: 2018-09-04 18:49:01.692660

"""
from alembic import op
from ckan.migration import skip_based_on_legacy_engine_version
revision = 'edcf3b8c3c1b'
down_revision = 'ecaa8b38782f'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    pass

def downgrade():
    if False:
        print('Hello World!')
    pass