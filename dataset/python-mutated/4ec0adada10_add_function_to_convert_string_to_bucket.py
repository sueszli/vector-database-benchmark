"""
Add function to convert string to bucket

Revision ID: 4ec0adada10
Revises: 9177113533
Create Date: 2015-09-06 19:32:50.438462
"""
from alembic import op
revision = '4ec0adada10'
down_revision = '9177113533'

def upgrade():
    if False:
        print('Hello World!')
    op.execute("\n        CREATE FUNCTION sitemap_bucket(text) RETURNS text AS $$\n                SELECT substring(\n                    encode(digest($1, 'sha512'), 'hex')\n                    from 1\n                    for 1\n                )\n            $$\n            LANGUAGE SQL\n            IMMUTABLE\n            RETURNS NULL ON NULL INPUT;\n    ")

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('DROP FUNCTION sitemap_bucket(text)')