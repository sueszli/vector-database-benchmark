"""idaddedtable

Revision ID: 05fbdb7027e7
Revises: b57eda13002b
Create Date: 2023-01-18 23:40:25.619057

"""
from alembic import op
import sqlalchemy as sa
revision = '05fbdb7027e7'
down_revision = 'b57eda13002b'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    pass

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_index(op.f('ix_product_key'), table_name='product')
    op.drop_table('product')