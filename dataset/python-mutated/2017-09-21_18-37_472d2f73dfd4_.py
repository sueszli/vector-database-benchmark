"""empty message

Revision ID: 472d2f73dfd4
Revises: ('19a814813610', 'a9c47e2c1547')
Create Date: 2017-09-21 18:37:30.844196

"""
revision = '472d2f73dfd4'
down_revision = ('19a814813610', 'a9c47e2c1547')
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    pass

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass