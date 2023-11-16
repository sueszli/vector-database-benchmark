"""Merge branching schema migrations

Revision ID: ab6e402016c7
Revises: c5b150e99eda
Create Date: 2022-09-16 10:44:04.663076

"""
import sqlalchemy as sa
from alembic import op
revision = 'ab6e402016c7'
down_revision = 'c5b150e99eda'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    pass

def downgrade():
    if False:
        i = 10
        return i + 15
    pass