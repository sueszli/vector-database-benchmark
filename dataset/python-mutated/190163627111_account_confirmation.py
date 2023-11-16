"""account confirmation

Revision ID: 190163627111
Revises: 456a945560f6
Create Date: 2013-12-29 02:58:45.577428

"""
revision = '190163627111'
down_revision = '456a945560f6'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        return 10
    op.add_column('users', sa.Column('confirmed', sa.Boolean(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('users', 'confirmed')