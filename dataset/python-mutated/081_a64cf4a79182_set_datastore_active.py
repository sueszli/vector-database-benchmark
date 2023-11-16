"""081 Set datastore active

Revision ID: a64cf4a79182
Revises: 8224d872c64f
Create Date: 2018-09-04 18:49:16.896531

"""
from alembic import op
from ckan.migration import skip_based_on_legacy_engine_version
revision = 'a64cf4a79182'
down_revision = '8224d872c64f'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    pass

def downgrade():
    if False:
        while True:
            i = 10
    pass