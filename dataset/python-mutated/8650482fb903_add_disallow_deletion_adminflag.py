"""
Add disallow-deletion AdminFlag

Revision ID: 8650482fb903
Revises: 34b18e18775c
Create Date: 2019-08-23 13:29:17.110252
"""
from alembic import op
revision = '8650482fb903'
down_revision = '34b18e18775c'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute("\n        INSERT INTO admin_flags(id, description, enabled, notify)\n        VALUES (\n            'disallow-deletion',\n            'Disallow ALL project and release deletions',\n            FALSE,\n            FALSE\n        )\n    ")

def downgrade():
    if False:
        while True:
            i = 10
    op.execute("DELETE FROM admin_flags WHERE id = 'disallow-deletion'")