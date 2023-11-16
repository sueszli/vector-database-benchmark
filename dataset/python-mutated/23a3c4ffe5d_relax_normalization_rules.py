"""
relax normalization rules

Revision ID: 23a3c4ffe5d
Revises: 91508cc5c2
Create Date: 2015-06-04 22:44:16.490470
"""
from alembic import op
revision = '23a3c4ffe5d'
down_revision = '91508cc5c2'

def upgrade():
    if False:
        while True:
            i = 10
    op.execute('DROP INDEX project_name_pep426_normalized')
    op.execute(" CREATE OR REPLACE FUNCTION normalize_pep426_name(text)\n            RETURNS text AS\n            $$\n                SELECT lower(regexp_replace($1, '(\\.|_)', '-', 'ig'))\n            $$\n            LANGUAGE SQL\n            IMMUTABLE\n            RETURNS NULL ON NULL INPUT;\n        ")

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute(" CREATE OR REPLACE FUNCTION normalize_pep426_name(text)\n            RETURNS text AS\n            $$\n                SELECT lower(\n                    regexp_replace(\n                        regexp_replace(\n                            regexp_replace($1, '(\\.|_)', '-', 'ig'),\n                            '(1|l|I)', '1', 'ig'\n                        ),\n                        '(0|0)', '0', 'ig'\n                    )\n                )\n            $$\n            LANGUAGE SQL\n            IMMUTABLE\n            RETURNS NULL ON NULL INPUT;\n        ")
    op.execute(' CREATE UNIQUE INDEX project_name_pep426_normalized\n            ON packages\n            (normalize_pep426_name(name))\n        ')