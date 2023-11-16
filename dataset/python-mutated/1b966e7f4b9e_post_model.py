"""post model

Revision ID: 1b966e7f4b9e
Revises: 198b0eebcf9
Create Date: 2013-12-31 00:00:14.700591

"""
revision = '1b966e7f4b9e'
down_revision = '198b0eebcf9'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        print('Hello World!')
    op.create_table('posts', sa.Column('id', sa.Integer(), nullable=False), sa.Column('body', sa.Text(), nullable=True), sa.Column('timestamp', sa.DateTime(), nullable=True), sa.Column('author_id', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['author_id'], ['users.id']), sa.PrimaryKeyConstraint('id'))
    op.create_index('ix_posts_timestamp', 'posts', ['timestamp'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index('ix_posts_timestamp', 'posts')
    op.drop_table('posts')