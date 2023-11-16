"""Add anonymous column for block documents

Revision ID: 61c76ee09e02
Revises: 3a7c41d3b464
Create Date: 2022-06-13 10:42:34.183100

"""
import sqlalchemy as sa
from alembic import op
revision = '61c76ee09e02'
down_revision = '3a7c41d3b464'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('block_document', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_anonymous', sa.Boolean(), server_default='0', nullable=False))
        batch_op.create_index(batch_op.f('ix_block_document__is_anonymous'), ['is_anonymous'], unique=False)

def downgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('block_document', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_block_document__is_anonymous'))
        batch_op.drop_column('is_anonymous')