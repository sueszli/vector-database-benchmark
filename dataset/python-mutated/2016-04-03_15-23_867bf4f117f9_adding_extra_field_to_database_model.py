"""Adding extra field to Database model

Revision ID: 867bf4f117f9
Revises: fee7b758c130
Create Date: 2016-04-03 15:23:20.280841

"""
revision = '867bf4f117f9'
down_revision = 'fee7b758c130'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    op.add_column('dbs', sa.Column('extra', sa.Text(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('dbs', 'extra')