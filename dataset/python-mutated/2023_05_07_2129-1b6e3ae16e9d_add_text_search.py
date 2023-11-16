"""add text search

Revision ID: 1b6e3ae16e9d
Revises: 9db92d504f64
Create Date: 2023-05-07 21:29:35.545612
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '1b6e3ae16e9d'
down_revision = '9db92d504f64'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.add_column('message', sa.Column('search_vector', postgresql.TSVECTOR(), nullable=True))
    op.create_index('idx_search_vector', 'message', ['search_vector'], postgresql_using='gin')

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_index('idx_search_vector', 'message')
    op.drop_column('message', 'search_vector')