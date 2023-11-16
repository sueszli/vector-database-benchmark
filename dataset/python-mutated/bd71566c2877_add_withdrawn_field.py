"""
add withdrawn field

Revision ID: bd71566c2877
Revises: 90f6ee9298db
Create Date: 2022-10-26 19:54:21.291985
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = 'bd71566c2877'
down_revision = '90f6ee9298db'

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('vulnerabilities', sa.Column('withdrawn', postgresql.TIMESTAMP(), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('vulnerabilities', 'withdrawn')