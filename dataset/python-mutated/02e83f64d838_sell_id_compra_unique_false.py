"""Sell id_compra unique false

Revision ID: 02e83f64d838
Revises: 683206855391
Create Date: 2023-07-23 18:04:39.458159

"""
from alembic import op
import sqlalchemy as sa
revision = '02e83f64d838'
down_revision = '683206855391'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    pass

def downgrade() -> None:
    if False:
        return 10
    pass