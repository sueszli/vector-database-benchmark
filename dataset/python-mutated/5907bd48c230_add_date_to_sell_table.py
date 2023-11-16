"""Add date to sell table

Revision ID: 5907bd48c230
Revises: 0178ee4aa404
Create Date: 2023-07-23 17:13:46.908607

"""
from alembic import op
import sqlalchemy as sa
revision = '5907bd48c230'
down_revision = '0178ee4aa404'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.add_column('sells', sa.Column('date', sa.DateTime(), nullable=True))

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_column('sells', 'date')