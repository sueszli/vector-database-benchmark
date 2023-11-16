"""Code to String

Revision ID: 6f9da26137b5
Revises: ed9067f0f7d6
Create Date: 2023-01-31 23:19:21.087424

"""
from alembic import op
import sqlalchemy as sa
revision = '6f9da26137b5'
down_revision = 'ed9067f0f7d6'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    pass

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    pass