"""add summary to resource

Revision ID: c02f3d759bf3
Revises: 1d54db311055
Create Date: 2023-06-27 05:07:29.016704

"""
from alembic import op
import sqlalchemy as sa
revision = 'c02f3d759bf3'
down_revision = 'c5c19944c90c'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.add_column('resources', sa.Column('summary', sa.Text(), nullable=True))

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_column('resources', 'summary')