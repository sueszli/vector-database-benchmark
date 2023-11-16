"""Add index to tagged_object

Revision ID: def97f26fdfb
Revises: d6ffdf31bdd4
Create Date: 2019-07-11 19:02:38.768324

"""
revision = 'def97f26fdfb'
down_revision = '190188938582'
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    op.create_index(op.f('ix_tagged_object_object_id'), 'tagged_object', ['object_id'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index(op.f('ix_tagged_object_object_id'), table_name='tagged_object')