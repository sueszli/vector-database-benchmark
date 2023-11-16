"""Sells Table

Revision ID: 5dcb17c7c713
Revises: fad64b75dcdf
Create Date: 2023-07-23 17:09:19.505581

"""
from alembic import op
import sqlalchemy as sa
revision = '5dcb17c7c713'
down_revision = 'fad64b75dcdf'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    pass