"""
Create a Normalize Function for PEP 426 names.

Revision ID: 20f4dbe11e9
Revises: 111d8fc0443
Create Date: 2015-04-04 23:29:58.373217
"""
from alembic import op
revision = '20f4dbe11e9'
down_revision = '111d8fc0443'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.execute("\n        CREATE FUNCTION normalize_pep426_name(text) RETURNS text AS $$\n                SELECT lower(\n                    regexp_replace(\n                        regexp_replace(\n                            regexp_replace($1, '(\\.|_)', '-', 'ig'),\n                            '(1|l|I)', '1', 'ig'\n                        ),\n                        '(0|0)', '0', 'ig'\n                    )\n                )\n            $$\n            LANGUAGE SQL\n            IMMUTABLE\n            RETURNS NULL ON NULL INPUT;\n    ")

def downgrade():
    if False:
        i = 10
        return i + 15
    op.execute('DROP FUNCTION normalize_pep426_name(text)')