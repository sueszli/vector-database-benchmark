"""
Drop JournalEntry.submitted_from

Revision ID: 60e6b0dd0f47
Revises: d738a238d781
Create Date: 2023-05-25 18:30:46.332534
"""
import sqlalchemy as sa
from alembic import op
revision = '60e6b0dd0f47'
down_revision = 'd738a238d781'

def upgrade():
    if False:
        return 10
    op.drop_column('journals', 'submitted_from')

def downgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('journals', sa.Column('submitted_from', sa.TEXT(), autoincrement=False, nullable=True))