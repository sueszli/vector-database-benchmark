"""
Add a function to get the array index

Revision ID: 6a03266b2d
Revises: 3bc5176b880
Create Date: 2015-11-18 18:29:57.554319
"""
from alembic import op
revision = '6a03266b2d'
down_revision = '3bc5176b880'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute(' CREATE FUNCTION array_idx(anyarray, anyelement)\n            RETURNS INT AS\n            $$\n                SELECT i FROM (\n                    SELECT generate_series(array_lower($1,1),array_upper($1,1))\n                ) g(i)\n                WHERE $1[i] = $2\n                LIMIT 1;\n            $$ LANGUAGE SQL IMMUTABLE;\n        ')

def downgrade():
    if False:
        print('Hello World!')
    op.execute('DROP FUNCTION array_idx')