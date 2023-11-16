"""
Add disallow-new-upload AdminFlag

Revision ID: ee4c59b2ef3a
Revises: 8650482fb903
Create Date: 2019-08-23 22:34:29.180163
"""
from alembic import op
revision = 'ee4c59b2ef3a'
down_revision = '8650482fb903'

def upgrade():
    if False:
        print('Hello World!')
    op.execute("\n        INSERT INTO admin_flags(id, description, enabled, notify)\n        VALUES (\n            'disallow-new-upload',\n            'Disallow ALL new uploads',\n            FALSE,\n            FALSE\n        )\n    ")

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute("DELETE FROM admin_flags WHERE id = 'disallow-new-upload'")