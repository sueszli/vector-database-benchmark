"""empty message

Revision ID: 8e80a26a31db
Revises: 2591d77e9831
Create Date: 2016-01-13 20:24:45.256437

"""
revision = '8e80a26a31db'
down_revision = '2591d77e9831'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('url', sa.Column('created_on', sa.DateTime(), nullable=False), sa.Column('changed_on', sa.DateTime(), nullable=False), sa.Column('id', sa.Integer(), nullable=False), sa.Column('url', sa.Text(), nullable=True), sa.Column('created_by_fk', sa.Integer(), nullable=True), sa.Column('changed_by_fk', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['changed_by_fk'], ['ab_user.id']), sa.ForeignKeyConstraint(['created_by_fk'], ['ab_user.id']), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('url')