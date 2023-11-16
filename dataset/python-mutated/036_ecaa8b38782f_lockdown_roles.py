"""036 Lockdown roles

Revision ID: ecaa8b38782f
Revises: 81148ccebd6c
Create Date: 2018-09-04 18:49:01.359019

"""
from alembic import op
from ckan.migration import skip_based_on_legacy_engine_version
revision = 'ecaa8b38782f'
down_revision = '81148ccebd6c'
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
        return 10
    pass