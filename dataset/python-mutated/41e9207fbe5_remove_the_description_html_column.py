"""
Remove the description_html column

Revision ID: 41e9207fbe5
Revises: 49b93c346db
Create Date: 2015-06-03 19:44:43.269987
"""
from alembic import op
revision = '41e9207fbe5'
down_revision = '49b93c346db'

def upgrade():
    if False:
        print('Hello World!')
    op.drop_column('releases', 'description_html')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    raise RuntimeError(f'Cannot downgrade past {revision!r}')