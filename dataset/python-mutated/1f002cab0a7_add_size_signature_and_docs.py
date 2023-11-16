"""
add size, signature, and docs

Revision ID: 1f002cab0a7
Revises: 283c68f2ab2
Create Date: 2015-06-02 23:50:02.029186
"""
import sqlalchemy as sa
from alembic import op
revision = '1f002cab0a7'
down_revision = '283c68f2ab2'

def upgrade():
    if False:
        return 10
    op.add_column('packages', sa.Column('has_docs', sa.Boolean(), nullable=True))
    op.add_column('release_files', sa.Column('has_signature', sa.Boolean(), nullable=True))
    op.add_column('release_files', sa.Column('size', sa.Integer(), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('release_files', 'size')
    op.drop_column('release_files', 'has_signature')
    op.drop_column('packages', 'has_docs')