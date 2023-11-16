"""drop_column_allow_multi_schema_metadata_fetch


Revision ID: 291f024254b5
Revises: 6d3c6f9d665d
Create Date: 2022-08-31 19:30:33.665025

"""
revision = '291f024254b5'
down_revision = '6d3c6f9d665d'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('dbs') as batch_op:
        batch_op.drop_column('allow_multi_schema_metadata_fetch')

def downgrade():
    if False:
        while True:
            i = 10
    op.add_column('dbs', sa.Column('allow_multi_schema_metadata_fetch', sa.Boolean(), nullable=True, default=True))