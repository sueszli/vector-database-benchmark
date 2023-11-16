"""Add Sells Table

Revision ID: 0178ee4aa404
Revises: 5dcb17c7c713
Create Date: 2023-07-23 17:11:55.420536

"""
from alembic import op
import sqlalchemy as sa
revision = '0178ee4aa404'
down_revision = '5dcb17c7c713'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.create_table('sells', sa.Column('id_compra', sa.Integer(), nullable=False), sa.Column('id_producto', sa.String(), nullable=True), sa.Column('cantidad', sa.Float(), nullable=True), sa.ForeignKeyConstraint(['id_producto'], ['product.key']), sa.PrimaryKeyConstraint('id_compra'))

def downgrade() -> None:
    if False:
        return 10
    op.drop_table('sells')