"""045 User name unique

Revision ID: 54e3f155d945
Revises: 4190eeeb8d73
Create Date: 2018-09-04 18:49:04.437304

"""
from alembic import op
from ckan.migration import skip_based_on_legacy_engine_version
revision = '54e3f155d945'
down_revision = '4190eeeb8d73'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    op.create_unique_constraint('user_name_key', 'user', ['name'])

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_constraint('user_name_key', 'user')