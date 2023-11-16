"""052 Update member capacities

Revision ID: ba693d64c6d7
Revises: a4fb0d85ced6
Create Date: 2018-09-04 18:49:06.885179

"""
from alembic import op
from ckan.migration import skip_based_on_legacy_engine_version
revision = 'ba693d64c6d7'
down_revision = 'a4fb0d85ced6'
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