"""Add block_type_name to block document

Revision ID: cef24af2ec34
Revises: f3165ae0a213
Create Date: 2023-10-30 07:50:26.414043

"""
import sqlalchemy as sa
from alembic import op
revision = 'cef24af2ec34'
down_revision = 'f3165ae0a213'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('block_document', schema=None) as batch_op:
        batch_op.add_column(sa.Column('block_type_name', sa.String(), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('block_document', schema=None) as batch_op:
        batch_op.drop_column('block_type_name')