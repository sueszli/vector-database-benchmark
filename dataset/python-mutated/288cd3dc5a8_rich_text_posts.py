"""rich text posts

Revision ID: 288cd3dc5a8
Revises: 1b966e7f4b9e
Create Date: 2013-12-31 03:25:13.286503

"""
revision = '288cd3dc5a8'
down_revision = '1b966e7f4b9e'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('posts', sa.Column('body_html', sa.Text(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('posts', 'body_html')