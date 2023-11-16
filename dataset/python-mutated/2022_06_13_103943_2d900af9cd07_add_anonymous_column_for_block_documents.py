"""Add anonymous column for block documents

Revision ID: 2d900af9cd07
Revises: 84892301571a
Create Date: 2022-06-13 10:39:43.872563

"""
import sqlalchemy as sa
from alembic import op
revision = '2d900af9cd07'
down_revision = '84892301571a'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('block_document', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_anonymous', sa.Boolean(), server_default='0', nullable=False))
        batch_op.create_index(batch_op.f('ix_block_document__is_anonymous'), ['is_anonymous'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('block_document', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_block_document__is_anonymous'))
        batch_op.drop_column('is_anonymous')