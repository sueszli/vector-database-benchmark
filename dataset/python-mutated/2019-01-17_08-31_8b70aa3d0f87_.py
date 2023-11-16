"""empty message

Revision ID: 8b70aa3d0f87
Revises: ('fbd55e0f83eb', 'fb13d49b72f9')
Create Date: 2019-01-17 08:31:55.781032

"""
revision = '8b70aa3d0f87'
down_revision = ('fbd55e0f83eb', 'fb13d49b72f9')
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    pass

def downgrade():
    if False:
        i = 10
        return i + 15
    pass