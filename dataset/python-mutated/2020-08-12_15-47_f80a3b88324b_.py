"""empty message

Revision ID: f80a3b88324b
Revises: ('978245563a02', 'f120347acb39')
Create Date: 2020-08-12 15:47:56.580191

"""
revision = 'f80a3b88324b'
down_revision = ('978245563a02', 'f120347acb39')
import sqlalchemy as sa
from alembic import op

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