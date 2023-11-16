"""
ultranormalize_name function

Revision ID: d18d443f89f0
Revises: d582fb87b94c
Create Date: 2021-12-17 06:25:19.035417
"""
from alembic import op
revision = 'd18d443f89f0'
down_revision = 'd582fb87b94c'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute("\n        CREATE FUNCTION ultranormalize_name(text) RETURNS text AS $$\n                SELECT lower(\n                    regexp_replace(\n                        regexp_replace(\n                            regexp_replace($1, '(\\.|_|-)', '', 'ig'),\n                            '(l|L|i|I)', '1', 'ig'\n                        ),\n                        '(o|O)', '0', 'ig'\n                    )\n                )\n            $$\n            LANGUAGE SQL\n            IMMUTABLE\n            RETURNS NULL ON NULL INPUT;\n    ")
    op.execute(' CREATE INDEX project_name_ultranormalized\n            ON projects\n            (ultranormalize_name(name))\n        ')

def downgrade():
    if False:
        i = 10
        return i + 15
    op.execute('DROP INDEX project_name_ultranormalized')
    op.execute('DROP FUNCTION ultranormalize_name(text)')