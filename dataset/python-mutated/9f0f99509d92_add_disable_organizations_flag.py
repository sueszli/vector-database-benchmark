"""
Add disable-organizations AdminFlag

Revision ID: 9f0f99509d92
Revises: 4a985d158c3c
Create Date: 2022-04-18 02:04:40.318843
"""
from alembic import op
revision = '9f0f99509d92'
down_revision = '4a985d158c3c'

def upgrade():
    if False:
        print('Hello World!')
    op.execute("\n        INSERT INTO admin_flags(id, description, enabled, notify)\n        VALUES (\n            'disable-organizations',\n            'Disallow ALL functionality for Organizations',\n            TRUE,\n            FALSE\n        )\n    ")

def downgrade():
    if False:
        print('Hello World!')
    op.execute("DELETE FROM admin_flags WHERE id = 'disable-organizations'")