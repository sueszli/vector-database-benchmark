"""empty message

Revision ID: 1b1938d5587a
Revises: aa7e3a012836, 0eaf40410361
Create Date: 2022-06-08 10:00:11.817097

"""
import sqlalchemy as sa
from alembic import op
revision = '1b1938d5587a'
down_revision = ('aa7e3a012836', '0eaf40410361')
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    pass

def downgrade():
    if False:
        i = 10
        return i + 15
    pass