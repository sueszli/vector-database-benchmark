"""user tokens

Revision ID: 834b1a697901
Revises: c81bac34faab
Create Date: 2017-11-05 18:41:07.996137

"""
from alembic import op
import sqlalchemy as sa
revision = '834b1a697901'
down_revision = 'c81bac34faab'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('user', sa.Column('token', sa.String(length=32), nullable=True))
    op.add_column('user', sa.Column('token_expiration', sa.DateTime(), nullable=True))
    op.create_index(op.f('ix_user_token'), 'user', ['token'], unique=True)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index(op.f('ix_user_token'), table_name='user')
    op.drop_column('user', 'token_expiration')
    op.drop_column('user', 'token')