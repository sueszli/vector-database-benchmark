"""Adding itemauditscores table

Revision ID: 67ea2aac5ea0
Revises: 55725cc4bf25
Create Date: 2016-01-26 21:43:10.398048

"""
revision = '67ea2aac5ea0'
down_revision = '55725cc4bf25'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        return 10
    op.create_table('itemauditscores', sa.Column('id', sa.Integer(), nullable=False), sa.Column('technology', sa.String(length=128), nullable=False), sa.Column('method', sa.String(length=256), nullable=False), sa.Column('score', sa.Integer(), nullable=False), sa.Column('disabled', sa.Boolean(), nullable=True), sa.PrimaryKeyConstraint('id'))
    op.create_unique_constraint('tech_method_uc', 'itemauditscores', ['technology', 'method'])

def downgrade():
    if False:
        return 10
    op.drop_table('itemauditscores')