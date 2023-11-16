"""empty message

Revision ID: 6414e83d82b7
Revises: ('525c854f0005', 'f1f2d4af5b90')
Create Date: 2016-12-19 09:57:05.814013

"""
revision = '6414e83d82b7'
down_revision = ('525c854f0005', 'f1f2d4af5b90')
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    pass

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass