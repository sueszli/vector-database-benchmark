"""add_certifications_columns_to_slice

Revision ID: f9847149153d
Revises: 32646df09c64
Create Date: 2021-11-03 14:07:09.905194

"""
import sqlalchemy as sa
from alembic import op
revision = 'f9847149153d'
down_revision = '32646df09c64'

def upgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('slices') as batch_op:
        batch_op.add_column(sa.Column('certified_by', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('certification_details', sa.Text(), nullable=True))

def downgrade():
    if False:
        return 10
    with op.batch_alter_table('slices') as batch_op:
        batch_op.drop_column('certified_by')
        batch_op.drop_column('certification_details')