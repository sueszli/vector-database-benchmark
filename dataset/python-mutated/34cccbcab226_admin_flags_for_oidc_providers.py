"""
admin flags for oidc providers

Revision ID: 34cccbcab226
Revises: 5dcbd2bc748f
Create Date: 2023-06-06 02:29:32.374813
"""
from alembic import op
revision = '34cccbcab226'
down_revision = '5dcbd2bc748f'

def upgrade():
    if False:
        return 10
    op.execute("\n        INSERT INTO admin_flags(id, description, enabled, notify)\n        VALUES (\n            'disallow-github-oidc',\n            'Disallow the GitHub OIDC provider',\n            FALSE,\n            FALSE\n        )\n        ")
    op.execute("\n        INSERT INTO admin_flags(id, description, enabled, notify)\n        VALUES (\n            'disallow-google-oidc',\n            'Disallow the Google OIDC provider',\n            FALSE,\n            FALSE\n        )\n        ")

def downgrade():
    if False:
        print('Hello World!')
    op.execute("DELETE FROM admin_flags WHERE id = 'disallow-github-oidc'")
    op.execute("DELETE FROM admin_flags WHERE id = 'disallow-google-oidc'")