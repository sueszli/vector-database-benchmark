"""010 Add user about

Revision ID: a6f13bf14d0c
Revises: b739a48de5c4
Create Date: 2018-09-04 18:44:53.313230

"""
from alembic import op
import sqlalchemy as sa
from ckan.migration import skip_based_on_legacy_engine_version
revision = 'a6f13bf14d0c'
down_revision = 'b739a48de5c4'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    op.add_column('user', sa.Column('about', sa.Text))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('user', 'about')