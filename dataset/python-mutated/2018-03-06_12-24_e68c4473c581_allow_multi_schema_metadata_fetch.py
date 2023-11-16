"""allow_multi_schema_metadata_fetch

Revision ID: e68c4473c581
Revises: e866bd2d4976
Create Date: 2018-03-06 12:24:30.896293

"""
import sqlalchemy as sa
from alembic import op
revision = 'e68c4473c581'
down_revision = 'e866bd2d4976'

def upgrade():
    if False:
        return 10
    op.add_column('dbs', sa.Column('allow_multi_schema_metadata_fetch', sa.Boolean(), nullable=True, default=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('dbs', 'allow_multi_schema_metadata_fetch')