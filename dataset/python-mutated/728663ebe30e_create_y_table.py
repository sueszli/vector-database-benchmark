"""create y table

Revision ID: 728663ebe30e
Revises: 4f59069f433e
Create Date: 2021-06-07 16:36:54.388978

"""
from alembic import op
import sqlalchemy as sa
revision = u'728663ebe30e'
down_revision = u'4f59069f433e'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.create_table(u'example_database_migrations_y', sa.Column(u'yid', sa.Integer, primary_key=True))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_table(u'example_database_migrations_y')