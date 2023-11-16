"""
readd the unique constraint on pep426 normalization

Revision ID: 1ce6d45d7ef
Revises: 23a3c4ffe5d
Create Date: 2015-06-04 23:09:11.612200
"""
from alembic import op
revision = '1ce6d45d7ef'
down_revision = '23a3c4ffe5d'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute(' CREATE UNIQUE INDEX project_name_pep426_normalized\n            ON packages\n            (normalize_pep426_name(name))\n        ')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('DROP INDEX project_name_pep426_normalized')