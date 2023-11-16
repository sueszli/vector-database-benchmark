"""
Add pep440_is_prerelease

Revision ID: e7b09b5c089d
Revises: be4cf6b58557
Create Date: 2016-12-03 15:04:40.251609
"""
from alembic import op
revision = 'e7b09b5c089d'
down_revision = 'be4cf6b58557'

def upgrade():
    if False:
        print('Hello World!')
    op.execute("\n        CREATE FUNCTION pep440_is_prerelease(text) returns boolean as $$\n                SELECT lower($1) ~* '(a|b|rc|dev|alpha|beta|c|pre|preview)'\n            $$\n            LANGUAGE SQL\n            IMMUTABLE\n            RETURNS NULL ON NULL INPUT;\n    ")

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('DROP FUNCTION pep440_is_prerelease')