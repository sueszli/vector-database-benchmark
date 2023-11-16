"""054 Add resource created date

Revision ID: da21b38da4db
Revises: 9d051a099097
Create Date: 2018-09-04 18:49:07.555419

"""
from alembic import op
import sqlalchemy as sa
from ckan.migration import skip_based_on_legacy_engine_version
revision = 'da21b38da4db'
down_revision = '9d051a099097'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    op.add_column('resource', sa.Column('created', sa.TIMESTAMP))
    op.add_column('resource_revision', sa.Column('created', sa.TIMESTAMP))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('resource', 'created')
    op.drop_column('resource_revision', 'created')