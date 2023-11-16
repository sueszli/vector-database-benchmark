"""tracking_url

Revision ID: ca69c70ec99b
Revises: a65458420354
Create Date: 2017-07-26 20:09:52.606416

"""
revision = 'ca69c70ec99b'
down_revision = 'a65458420354'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('query', sa.Column('tracking_url', sa.Text(), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('query', 'tracking_url')