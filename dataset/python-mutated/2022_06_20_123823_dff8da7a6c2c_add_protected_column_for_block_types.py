"""Add protected column for block types

Revision ID: dff8da7a6c2c
Revises: 9e2a1c08c6f1
Create Date: 2022-06-20 12:38:23.657760

"""
import sqlalchemy as sa
from alembic import op
revision = 'dff8da7a6c2c'
down_revision = 'a205b458d997'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('block_type', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_protected', sa.Boolean(), server_default='0', nullable=False))

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('block_type', schema=None) as batch_op:
        batch_op.drop_column('is_protected')