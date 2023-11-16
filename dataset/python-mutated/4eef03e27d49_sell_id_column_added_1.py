"""Sell id column added 1

Revision ID: 4eef03e27d49
Revises: e5a5a807fc71
Create Date: 2023-07-23 18:10:58.143924

"""
from alembic import op
import sqlalchemy as sa
revision = '4eef03e27d49'
down_revision = 'e5a5a807fc71'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.create_table('sells', sa.Column('id', sa.Integer(), autoincrement=True, nullable=False), sa.Column('id_compra', sa.Integer(), nullable=False), sa.Column('id_producto', sa.String(), nullable=False), sa.Column('cantidad', sa.Float(), nullable=False), sa.Column('date', sa.DateTime(), nullable=False), sa.ForeignKeyConstraint(['id_producto'], ['product.key']), sa.PrimaryKeyConstraint('id'))

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_table('sells')