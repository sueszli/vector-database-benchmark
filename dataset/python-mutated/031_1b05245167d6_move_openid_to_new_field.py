"""031 Move openid to new_field

Revision ID: 1b05245167d6
Revises: b16cbf164c8a
Create Date: 2018-09-04 18:48:59.666938

"""
from alembic import op
from ckan.migration import skip_based_on_legacy_engine_version
revision = '1b05245167d6'
down_revision = 'b16cbf164c8a'
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