"""create x table

Revision ID: 4f59069f433e
Revises:
Create Date: 2021-06-07 16:36:45.509302

"""
from alembic import op
import sqlalchemy as sa
revision = u'4f59069f433e'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table(u'example_database_migrations_x', sa.Column(u'xid', sa.Integer, primary_key=True))

def downgrade():
    if False:
        return 10
    op.drop_table(u'example_database_migrations_x')