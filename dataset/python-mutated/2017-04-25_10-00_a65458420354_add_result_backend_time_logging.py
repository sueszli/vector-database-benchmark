"""add_result_backend_time_logging

Revision ID: a65458420354
Revises: 2fcdcb35e487
Create Date: 2017-04-25 10:00:58.053120

"""
import sqlalchemy as sa
from alembic import op
revision = 'a65458420354'
down_revision = '2fcdcb35e487'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('query', sa.Column('end_result_backend_time', sa.Numeric(precision=20, scale=6), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('query', 'end_result_backend_time')