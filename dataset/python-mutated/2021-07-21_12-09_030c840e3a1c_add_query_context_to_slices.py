"""Add query context to slices

Revision ID: 030c840e3a1c
Revises: 3317e9248280
Create Date: 2021-07-21 12:09:37.048337

"""
revision = '030c840e3a1c'
down_revision = '3317e9248280'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

def upgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('slices') as batch_op:
        batch_op.add_column(sa.Column('query_context', sa.Text(), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('slices') as batch_op:
        batch_op.drop_column('query_context')