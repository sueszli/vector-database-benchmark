"""empty message

Revision ID: fc480c87706c
Revises: ('4451805bbaa1', '1d9e835a84f9')
Create Date: 2018-07-22 11:50:54.174443

"""
revision = 'fc480c87706c'
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
        while True:
            i = 10
    pass