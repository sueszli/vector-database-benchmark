"""idadded

Revision ID: b57eda13002b
Revises: c8ba2e311e15
Create Date: 2023-01-18 23:39:25.289623

"""
from alembic import op
import sqlalchemy as sa
revision = 'b57eda13002b'
down_revision = 'c8ba2e311e15'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.add_column('product', sa.Column('id', sa.Integer(), nullable=True))
    op.create_unique_constraint(None, 'product', ['id'])

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_constraint(None, 'product', type_='unique')
    op.drop_column('product', 'id')