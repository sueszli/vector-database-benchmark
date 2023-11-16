"""drop access_request

Revision ID: 83e1abbe777f
Revises: ae58e1e58e5c
Create Date: 2023-06-01 13:13:18.147362

"""
revision = '83e1abbe777f'
down_revision = 'ae58e1e58e5c'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('access_request')

def downgrade():
    if False:
        return 10
    op.create_table('access_request', sa.Column('created_on', sa.DateTime(), nullable=True), sa.Column('changed_on', sa.DateTime(), nullable=True), sa.Column('id', sa.Integer(), nullable=False), sa.Column('datasource_type', sa.String(length=200), nullable=True), sa.Column('datasource_id', sa.Integer(), nullable=True), sa.Column('changed_by_fk', sa.Integer(), nullable=True), sa.Column('created_by_fk', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['changed_by_fk'], ['ab_user.id']), sa.ForeignKeyConstraint(['created_by_fk'], ['ab_user.id']), sa.PrimaryKeyConstraint('id'))