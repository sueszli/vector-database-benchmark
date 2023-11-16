"""075 Rename view plugins

Revision ID: 9cdc88c8896a
Revises: a4ca55f0f45e
Create Date: 2018-09-04 18:49:14.766120

"""
from alembic import op
from ckan.migration import skip_based_on_legacy_engine_version
revision = '9cdc88c8896a'
down_revision = 'a4ca55f0f45e'
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
        while True:
            i = 10
    pass