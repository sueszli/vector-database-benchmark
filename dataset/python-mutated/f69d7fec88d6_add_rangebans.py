"""add rangebans

Revision ID: f69d7fec88d6
Revises: 6cc823948c5a
Create Date: 2018-06-01 14:01:49.596007

"""
from alembic import op
import sqlalchemy as sa
revision = 'f69d7fec88d6'
down_revision = '6cc823948c5a'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.create_table('rangebans', sa.Column('id', sa.Integer(), nullable=False), sa.Column('cidr_string', sa.String(length=18), nullable=False), sa.Column('masked_cidr', sa.BigInteger(), nullable=False), sa.Column('mask', sa.BigInteger(), nullable=False), sa.Column('enabled', sa.Boolean(), nullable=False), sa.Column('temp', sa.DateTime(), nullable=True), sa.PrimaryKeyConstraint('id'))
    op.create_index(op.f('ix_rangebans_mask'), 'rangebans', ['mask'], unique=False)
    op.create_index(op.f('ix_rangebans_masked_cidr'), 'rangebans', ['masked_cidr'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index(op.f('ix_rangebans_masked_cidr'), table_name='rangebans')
    op.drop_index(op.f('ix_rangebans_mask'), table_name='rangebans')
    op.drop_table('rangebans')