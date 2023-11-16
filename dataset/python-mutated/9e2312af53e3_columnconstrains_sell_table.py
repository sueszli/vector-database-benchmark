"""ColumnConstrains Sell Table

Revision ID: 9e2312af53e3
Revises: 5907bd48c230
Create Date: 2023-07-23 17:15:31.323418

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision = '9e2312af53e3'
down_revision = '5907bd48c230'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.alter_column('sells', 'id_producto', existing_type=sa.VARCHAR(), nullable=False)
    op.alter_column('sells', 'cantidad', existing_type=postgresql.DOUBLE_PRECISION(precision=53), nullable=False)
    op.alter_column('sells', 'date', existing_type=postgresql.TIMESTAMP(), nullable=False)
    op.create_unique_constraint(None, 'sells', ['id_compra'])

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_constraint(None, 'sells', type_='unique')
    op.alter_column('sells', 'date', existing_type=postgresql.TIMESTAMP(), nullable=True)
    op.alter_column('sells', 'cantidad', existing_type=postgresql.DOUBLE_PRECISION(precision=53), nullable=True)
    op.alter_column('sells', 'id_producto', existing_type=sa.VARCHAR(), nullable=True)