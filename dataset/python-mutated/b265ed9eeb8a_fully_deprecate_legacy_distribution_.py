"""
Fully deprecate legacy distribution types

Revision ID: b265ed9eeb8a
Revises: c4cb2d15dada
Create Date: 2020-03-12 17:51:08.447903
"""
from alembic import op
revision = 'b265ed9eeb8a'
down_revision = 'c4cb2d15dada'

def upgrade():
    if False:
        while True:
            i = 10
    op.drop_column('projects', 'allow_legacy_files')

def downgrade():
    if False:
        while True:
            i = 10
    raise RuntimeError('Order No. 227 - Ни шагу назад!')