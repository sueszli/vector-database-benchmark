"""empty message

Revision ID: c829ff0b37d0
Revises: ('4451805bbaa1', '1d9e835a84f9')
Create Date: 2018-07-22 08:49:48.936117

"""
revision = 'c829ff0b37d0'
down_revision = ('4451805bbaa1', '1d9e835a84f9')
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    pass

def downgrade():
    if False:
        print('Hello World!')
    pass