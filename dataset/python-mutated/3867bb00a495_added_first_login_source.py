"""added_first_login_source

Revision ID: 3867bb00a495
Revises: 661ec8a4c32e
Create Date: 2023-09-15 02:06:24.006555

"""
from alembic import op
import sqlalchemy as sa
revision = '3867bb00a495'
down_revision = '661ec8a4c32e'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.add_column('users', sa.Column('first_login_source', sa.String(), nullable=True))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('users', 'first_login_source')