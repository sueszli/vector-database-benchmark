"""empty message

Revision ID: fbd55e0f83eb
Revises: ('7467e77870e4', 'de021a1ca60d')
Create Date: 2018-12-22 17:26:16.113317

"""
revision = 'fbd55e0f83eb'
down_revision = ('7467e77870e4', 'de021a1ca60d')
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