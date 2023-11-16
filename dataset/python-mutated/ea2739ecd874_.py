"""Ensures that account.identifier is unique.

Revision ID: ea2739ecd874
Revises: 5bd631a1b748
Create Date: 2017-09-29 09:16:09.436339

"""
revision = 'ea2739ecd874'
down_revision = '5bd631a1b748'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_unique_constraint(None, 'account', ['identifier'])

def downgrade():
    if False:
        print('Hello World!')
    op.drop_constraint(None, 'account', type_='unique')