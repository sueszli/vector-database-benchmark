"""idnotunique

Revision ID: a3456c9cab2c
Revises: 05fbdb7027e7
Create Date: 2023-01-18 23:42:39.977815

"""
from alembic import op
import sqlalchemy as sa
revision = 'a3456c9cab2c'
down_revision = '05fbdb7027e7'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_constraint('product_id_key', 'product', type_='unique')

def downgrade() -> None:
    if False:
        print('Hello World!')
    op.create_unique_constraint('product_id_key', 'product', ['id'])