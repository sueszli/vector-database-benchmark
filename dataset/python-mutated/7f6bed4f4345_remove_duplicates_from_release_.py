"""
Remove duplicates from release_classifiers

Revision ID: 7f6bed4f4345
Revises: a8ebe73ccaf2
Create Date: 2023-08-16 22:56:54.898269
"""
from alembic import op
revision = '7f6bed4f4345'
down_revision = 'a8ebe73ccaf2'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('SET statement_timeout = 60000')
    op.execute('SET lock_timeout = 60000')
    op.execute('\n        DELETE FROM release_classifiers a USING (\n            SELECT MIN(ctid) as ctid, release_id, trove_id\n            FROM release_classifiers\n            GROUP BY release_id, trove_id HAVING COUNT(*) > 1\n            LIMIT 4453 -- 4453 is the number of duplicates in production\n        ) b\n        WHERE a.release_id = b.release_id\n        AND a.trove_id = b.trove_id\n        AND a.ctid <> b.ctid;\n        ')

def downgrade():
    if False:
        while True:
            i = 10
    pass