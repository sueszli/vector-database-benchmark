"""merge point

Revision ID: 9d8a8d575284
Revises: ('8b841273bec3', 'b0d0249074e4')
Create Date: 2022-04-06 14:10:40.433050

"""
revision = '9d8a8d575284'
down_revision = ('8b841273bec3', 'b0d0249074e4')
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    pass

def downgrade():
    if False:
        while True:
            i = 10
    pass