"""empty message

Revision ID: e3970889f38e
Revises: ('4451805bbaa1', '1d9e835a84f9')
Create Date: 2018-07-22 09:32:36.986561

"""
revision = 'e3970889f38e'
down_revision = ('4451805bbaa1', '1d9e835a84f9')
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