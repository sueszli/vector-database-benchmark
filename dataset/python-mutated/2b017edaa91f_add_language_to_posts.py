"""add language to posts

Revision ID: 2b017edaa91f
Revises: ae346256b650
Create Date: 2017-10-04 22:48:34.494465

"""
from alembic import op
import sqlalchemy as sa
revision = '2b017edaa91f'
down_revision = 'ae346256b650'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('post', sa.Column('language', sa.String(length=5), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('post', 'language')