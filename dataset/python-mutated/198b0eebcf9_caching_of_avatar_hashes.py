"""caching of avatar hashes

Revision ID: 198b0eebcf9
Revises: d66f086b258
Create Date: 2014-02-04 09:10:02.245503

"""
revision = '198b0eebcf9'
down_revision = 'd66f086b258'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('users', sa.Column('avatar_hash', sa.String(length=32), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('users', 'avatar_hash')