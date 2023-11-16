"""Merge heads

Revision ID: 01f11a3541cc
Revises: 3433a9040ff4
Create Date: 2022-09-14 09:38:37.284015

"""
import sqlalchemy as sa
from alembic import op
revision = '01f11a3541cc'
down_revision = '3433a9040ff4'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    pass

def downgrade():
    if False:
        i = 10
        return i + 15
    pass