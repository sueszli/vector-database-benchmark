"""Sells id

Revision ID: 683206855391
Revises: 73ac87195b7a
Create Date: 2023-07-23 17:51:59.349105

"""
from alembic import op
import sqlalchemy as sa
revision = '683206855391'
down_revision = '73ac87195b7a'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    op.drop_constraint('sells_id_compra_key', 'sells', type_='unique')

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.create_unique_constraint('sells_id_compra_key', 'sells', ['id_compra'])