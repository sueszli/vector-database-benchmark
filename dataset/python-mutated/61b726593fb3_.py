"""Remove SSHKey.key 

Revision ID: 61b726593fb3
Revises: 0482abd84ff2
Create Date: 2022-12-22 14:18:54.208321

"""
import sqlalchemy as sa
from alembic import op
revision = '61b726593fb3'
down_revision = '0482abd84ff2'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('ssh_keys', 'key')

def downgrade():
    if False:
        print('Hello World!')
    op.add_column('ssh_keys', sa.Column('key', sa.VARCHAR(), autoincrement=False, nullable=False))