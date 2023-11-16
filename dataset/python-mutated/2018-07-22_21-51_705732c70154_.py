"""empty message

Revision ID: 705732c70154
Revises: ('4451805bbaa1', '1d9e835a84f9')
Create Date: 2018-07-22 21:51:19.235558

"""
revision = '705732c70154'
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
        i = 10
        return i + 15
    pass