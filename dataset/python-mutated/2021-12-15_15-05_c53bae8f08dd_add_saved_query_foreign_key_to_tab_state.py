"""add_saved_query_foreign_key_to_tab_state

Revision ID: c53bae8f08dd
Revises: bb38f40aa3ff
Create Date: 2021-12-15 15:05:21.845777
"""
revision = 'c53bae8f08dd'
down_revision = 'bb38f40aa3ff'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('tab_state') as batch_op:
        batch_op.add_column(sa.Column('saved_query_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('saved_query_id', 'saved_query', ['saved_query_id'], ['id'])

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('tab_state') as batch_op:
        batch_op.drop_constraint('saved_query_id', type_='foreignkey')
        batch_op.drop_column('saved_query_id')