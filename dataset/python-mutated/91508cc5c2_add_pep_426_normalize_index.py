"""
Add Index for normalized PEP 426 names which enforces uniqueness.

Revision ID: 91508cc5c2
Revises: 20f4dbe11e9
Create Date: 2015-04-04 23:55:27.024988
"""
from alembic import op
revision = '91508cc5c2'
down_revision = '20f4dbe11e9'

def upgrade():
    if False:
        print('Hello World!')
    op.execute('\n        CREATE UNIQUE INDEX project_name_pep426_normalized\n            ON packages\n            (normalize_pep426_name(name))\n    ')

def downgrade():
    if False:
        print('Hello World!')
    op.execute('DROP INDEX project_name_pep426_normalized')