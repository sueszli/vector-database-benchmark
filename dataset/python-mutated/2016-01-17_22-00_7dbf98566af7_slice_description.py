"""empty message

Revision ID: 7dbf98566af7
Revises: 8e80a26a31db
Create Date: 2016-01-17 22:00:23.640788

"""
revision = '7dbf98566af7'
down_revision = '8e80a26a31db'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('slices', sa.Column('description', sa.Text(), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('slices', 'description')