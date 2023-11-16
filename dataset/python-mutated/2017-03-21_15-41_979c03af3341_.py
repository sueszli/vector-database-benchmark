"""empty message

Revision ID: 979c03af3341
Revises: ('db527d8c4c78', 'ea033256294a')
Create Date: 2017-03-21 15:41:34.383808

"""
revision = '979c03af3341'
down_revision = ('db527d8c4c78', 'ea033256294a')
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    pass

def downgrade():
    if False:
        return 10
    pass