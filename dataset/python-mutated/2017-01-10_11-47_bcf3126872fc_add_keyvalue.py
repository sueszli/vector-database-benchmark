"""Add keyvalue table

Revision ID: bcf3126872fc
Revises: f18570e03440
Create Date: 2017-01-10 11:47:56.306938

"""
revision = 'bcf3126872fc'
down_revision = 'f18570e03440'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('keyvalue', sa.Column('id', sa.Integer(), nullable=False), sa.Column('value', sa.Text(), nullable=False), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('keyvalue')