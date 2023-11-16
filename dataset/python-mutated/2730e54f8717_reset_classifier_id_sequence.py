"""
Reset Classifier ID sequence

Revision ID: 2730e54f8717
Revises: 8fd3400c760f
Create Date: 2018-03-14 16:34:38.151300
"""
from alembic import op
revision = '2730e54f8717'
down_revision = '8fd3400c760f'

def upgrade():
    if False:
        return 10
    op.execute("\n        SELECT setval('trove_classifiers_id_seq', max(id))\n        FROM trove_classifiers;\n    ")

def downgrade():
    if False:
        print('Hello World!')
    pass