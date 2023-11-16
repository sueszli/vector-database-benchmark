"""Sell id Auto  Increment

Revision ID: 73ac87195b7a
Revises: 9e2312af53e3
Create Date: 2023-07-23 17:22:02.546938

"""
from alembic import op
import sqlalchemy as sa
revision = '73ac87195b7a'
down_revision = '9e2312af53e3'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

def downgrade() -> None:
    if False:
        while True:
            i = 10
    pass