"""add auth_method to person

Revision ID: 6368515778c5
Revises: cd7de470586e
Create Date: 2022-12-17 17:57:33.022549

"""
import sqlalchemy as sa
from alembic import op
revision = '6368515778c5'
down_revision = 'cd7de470586e'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.add_column('person', sa.Column('auth_method', sa.String(length=128), nullable=True))
    op.execute("UPDATE person SET auth_method = 'local'")
    op.alter_column('person', 'auth_method', nullable=False)

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_column('person', 'auth_method')