"""
redistribute sitemap buckets

Revision ID: c4cb2d15dada
Revises: d15f020ee3df
Create Date: 2020-04-07 16:59:56.333491
"""
from alembic import op
revision = 'c4cb2d15dada'
down_revision = 'd15f020ee3df'

def upgrade():
    if False:
        return 10
    op.execute("\n        CREATE OR REPLACE FUNCTION sitemap_bucket(text) RETURNS text AS $$\n                SELECT substring(\n                    encode(digest($1, 'sha512'), 'hex')\n                    from 1\n                    for 2\n                )\n            $$\n            LANGUAGE SQL\n            IMMUTABLE\n            RETURNS NULL ON NULL INPUT;\n    ")
    op.execute('\n        UPDATE users\n        SET sitemap_bucket = sitemap_bucket(username)\n        ')
    op.execute('\n        UPDATE projects\n        SET sitemap_bucket = sitemap_bucket(name)\n        ')

def downgrade():
    if False:
        return 10
    op.execute("\n        CREATE OR REPLACE FUNCTION sitemap_bucket(text) RETURNS text AS $$\n                SELECT substring(\n                    encode(digest($1, 'sha512'), 'hex')\n                    from 1\n                    for 1\n                )\n            $$\n            LANGUAGE SQL\n            IMMUTABLE\n            RETURNS NULL ON NULL INPUT;\n    ")
    op.execute('\n        UPDATE users\n        SET sitemap_bucket = sitemap_bucket(username)\n        ')
    op.execute('\n        UPDATE projects\n        SET sitemap_bucket = sitemap_bucket(name)\n        ')