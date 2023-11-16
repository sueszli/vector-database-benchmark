"""Sell id column added

Revision ID: e5a5a807fc71
Revises: 6e5731b30b6c
Create Date: 2023-07-23 18:09:40.603391

"""
from alembic import op
import sqlalchemy as sa
revision = 'e5a5a807fc71'
down_revision = '6e5731b30b6c'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.add_column('sells', sa.Column('id', sa.Integer(), autoincrement=True, nullable=False))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('sells', 'id')