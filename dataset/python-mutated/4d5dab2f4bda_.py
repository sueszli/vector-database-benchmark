"""empty message

Revision ID: 4d5dab2f4bda
Revises: c7eb1647cba3, 61b726593fb3
Create Date: 2023-01-19 16:14:47.465395

"""
import sqlalchemy as sa
from alembic import op
revision = '4d5dab2f4bda'
down_revision = ('c7eb1647cba3', '61b726593fb3')
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    pass

def downgrade():
    if False:
        i = 10
        return i + 15
    pass