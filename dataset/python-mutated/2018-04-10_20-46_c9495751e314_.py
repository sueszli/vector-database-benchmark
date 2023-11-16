"""empty message

Revision ID: c9495751e314
Revises: ('30bb17c0dc76', 'bf706ae5eb46')
Create Date: 2018-04-10 20:46:57.890773

"""
revision = 'c9495751e314'
down_revision = ('30bb17c0dc76', 'bf706ae5eb46')
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    pass

def downgrade():
    if False:
        return 10
    pass