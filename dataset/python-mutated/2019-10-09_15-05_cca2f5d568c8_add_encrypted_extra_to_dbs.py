"""add encrypted_extra to dbs

Revision ID: cca2f5d568c8
Revises: b6fa807eac07
Create Date: 2019-10-09 15:05:06.965042

"""
revision = 'cca2f5d568c8'
down_revision = 'b6fa807eac07'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('dbs', sa.Column('encrypted_extra', sa.Text(), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('dbs', 'encrypted_extra')