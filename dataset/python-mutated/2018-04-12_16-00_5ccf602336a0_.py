"""empty message

Revision ID: 5ccf602336a0
Revises: ('130915240929', 'c9495751e314')
Create Date: 2018-04-12 16:00:47.639218

"""
revision = '5ccf602336a0'
down_revision = ('130915240929', 'c9495751e314')
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