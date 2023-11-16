"""
Make File.path mandatory

Revision ID: f46672a776f1
Revises: 6ff880c36cd9
Create Date: 2016-01-07 13:17:25.942208
"""
from alembic import op
revision = 'f46672a776f1'
down_revision = '6ff880c36cd9'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute(" UPDATE release_files\n               SET path = concat_ws(\n                            '/',\n                            python_version,\n                            substring(name, 1, 1),\n                            name,\n                            filename\n                          )\n             WHERE path IS NULL\n        ")
    op.alter_column('release_files', 'path', nullable=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.alter_column('release_files', 'path', nullable=True)