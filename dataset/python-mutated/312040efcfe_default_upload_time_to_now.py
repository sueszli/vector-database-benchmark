"""
default upload_time to now

Revision ID: 312040efcfe
Revises: 57b1053998d
Create Date: 2015-06-13 01:44:23.445650
"""
import sqlalchemy as sa
from alembic import op
revision = '312040efcfe'
down_revision = '57b1053998d'

def upgrade():
    if False:
        print('Hello World!')
    op.alter_column('release_files', 'upload_time', server_default=sa.text('now()'))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.alter_column('release_files', 'upload_time', server_default=None)