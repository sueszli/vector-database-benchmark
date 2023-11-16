"""Adding accountpatternauditscore table

Revision ID: 6d2354fb841c
Revises: 67ea2aac5ea0
Create Date: 2016-06-21 19:58:12.949279

"""
revision = '6d2354fb841c'
down_revision = '67ea2aac5ea0'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('accountpatternauditscore', sa.Column('id', sa.Integer(), nullable=False), sa.Column('account_type', sa.String(length=80), nullable=False), sa.Column('account_field', sa.String(length=128), nullable=False), sa.Column('account_pattern', sa.String(length=128), nullable=False), sa.Column('score', sa.Integer(), nullable=False), sa.Column('itemauditscores_id', sa.Integer(), nullable=False), sa.ForeignKeyConstraint(['itemauditscores_id'], ['itemauditscores.id']), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_table('accountpatternauditscore')