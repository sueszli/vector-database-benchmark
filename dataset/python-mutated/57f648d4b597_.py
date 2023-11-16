"""Add auditorsettings

Revision ID: 57f648d4b597
Revises: 2705e6e13a8f
Create Date: 2015-01-30 22:32:18.420819

"""
revision = '57f648d4b597'
down_revision = '2705e6e13a8f'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        return 10
    op.create_table('auditorsettings', sa.Column('id', sa.Integer(), nullable=False), sa.Column('tech_id', sa.Integer(), nullable=True), sa.Column('notes', sa.String(length=512), nullable=True), sa.Column('account_id', sa.Integer(), nullable=True), sa.Column('disabled', sa.Boolean(), nullable=False), sa.Column('issue', sa.String(length=512), nullable=False), sa.ForeignKeyConstraint(['account_id'], ['account.id']), sa.ForeignKeyConstraint(['tech_id'], ['technology.id']), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_table('auditorsettings')