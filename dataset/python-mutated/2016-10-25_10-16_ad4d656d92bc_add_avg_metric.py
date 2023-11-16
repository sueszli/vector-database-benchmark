"""Add avg() to default metrics

Revision ID: ad4d656d92bc
Revises: b46fa1b0b39e
Create Date: 2016-10-25 10:16:39.871078

"""
revision = 'ad4d656d92bc'
down_revision = '7e3ddad2a00b'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('columns', sa.Column('avg', sa.Boolean(), nullable=True))
    op.add_column('table_columns', sa.Column('avg', sa.Boolean(), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('columns') as batch_op:
        batch_op.drop_column('avg')
    with op.batch_alter_table('table_columns') as batch_op:
        batch_op.drop_column('avg')