"""Sell id_compra unique false 1

Revision ID: 6e5731b30b6c
Revises: 02e83f64d838
Create Date: 2023-07-23 18:06:58.079351

"""
from alembic import op
import sqlalchemy as sa
revision = '6e5731b30b6c'
down_revision = '02e83f64d838'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    pass

def downgrade() -> None:
    if False:
        return 10
    pass