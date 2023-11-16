"""
Normalize runs of characters to a single character

Revision ID: 3af8d0006ba
Revises: 5ff0c99c94
Create Date: 2015-08-17 21:05:51.699639
"""
from alembic import op
revision = '3af8d0006ba'
down_revision = '5ff0c99c94'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute(" CREATE OR REPLACE FUNCTION normalize_pep426_name(text)\n            RETURNS text AS\n            $$\n                SELECT lower(regexp_replace($1, '(\\.|_|-)+', '-', 'ig'))\n            $$\n            LANGUAGE SQL\n            IMMUTABLE\n            RETURNS NULL ON NULL INPUT;\n        ")
    op.execute('REINDEX INDEX project_name_pep426_normalized')

def downgrade():
    if False:
        return 10
    op.execute(" CREATE OR REPLACE FUNCTION normalize_pep426_name(text)\n            RETURNS text AS\n            $$\n                SELECT lower(regexp_replace($1, '(\\.|_)', '-', 'ig'))\n            $$\n            LANGUAGE SQL\n            IMMUTABLE\n            RETURNS NULL ON NULL INPUT;\n        ")
    op.execute('REINDEX INDEX project_name_pep426_normalized')