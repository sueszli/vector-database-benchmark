"""050 Term translation table

Revision ID: 01a6b058cb7f
Revises: e0c06c2177b5
Create Date: 2018-09-04 18:49:06.143050

"""
from alembic import op
import sqlalchemy as sa
from ckan.migration import skip_based_on_legacy_engine_version
revision = '01a6b058cb7f'
down_revision = 'e0c06c2177b5'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    op.create_table('term_translation', sa.Column('term', sa.UnicodeText, nullable=False), sa.Column('term_translation', sa.UnicodeText, nullable=False), sa.Column('lang_code', sa.UnicodeText, nullable=False))
    op.create_index('term_lang', 'term_translation', ['term', 'lang_code'])
    op.create_index('term', 'term_translation', ['term'])

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('term_translation')