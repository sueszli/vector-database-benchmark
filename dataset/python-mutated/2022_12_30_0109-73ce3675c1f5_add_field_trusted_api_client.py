"""add field trusted api client

Revision ID: 73ce3675c1f5
Revises: 464ec4667aae
Create Date: 2022-12-30 01:09:06.446020

"""
import sqlalchemy as sa
from alembic import op
revision = '73ce3675c1f5'
down_revision = '464ec4667aae'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.add_column('api_client', sa.Column('trusted', sa.Boolean(), server_default=sa.text('false'), nullable=False))

def downgrade() -> None:
    if False:
        return 10
    op.drop_column('api_client', 'trusted')