"""local_llms

Revision ID: 9270eb5a8475
Revises: 3867bb00a495
Create Date: 2023-10-04 09:26:33.865424

"""
from alembic import op
import sqlalchemy as sa
revision = '9270eb5a8475'
down_revision = '3867bb00a495'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.add_column('models', sa.Column('context_length', sa.Integer(), nullable=True))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('models', 'context_length')