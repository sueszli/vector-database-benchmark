"""Adding a fixed flag to the issue table

Revision ID: 4ac52090a637
Revises: daee17da2abd
Create Date: 2017-09-14 23:24:40.967949

"""
revision = '4ac52090a637'
down_revision = 'daee17da2abd'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('itemaudit', sa.Column('fixed', sa.Boolean(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('itemaudit', 'fixed')