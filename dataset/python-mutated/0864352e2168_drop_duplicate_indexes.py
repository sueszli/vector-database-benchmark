"""
Drop duplicate indexes

Revision ID: 0864352e2168
Revises: 6a6eb0a95603
Create Date: 2018-08-15 20:27:08.429077
"""
from alembic import op
revision = '0864352e2168'
down_revision = '6a6eb0a95603'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.drop_index('accounts_email_email_like', table_name='accounts_email')
    op.drop_index('journals_id_idx', table_name='journals')
    op.drop_index('trove_class_class_idx', table_name='trove_classifiers')
    op.drop_index('trove_class_id_idx', table_name='trove_classifiers')

def downgrade():
    if False:
        while True:
            i = 10
    op.create_index('trove_class_id_idx', 'trove_classifiers', ['id'], unique=False)
    op.create_index('trove_class_class_idx', 'trove_classifiers', ['classifier'], unique=False)
    op.create_index('journals_id_idx', 'journals', ['id'], unique=False)
    op.create_index('accounts_email_email_like', 'accounts_email', ['email'], unique=False)