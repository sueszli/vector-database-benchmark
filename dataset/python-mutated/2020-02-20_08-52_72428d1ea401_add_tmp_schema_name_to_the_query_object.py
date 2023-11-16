"""Add tmp_schema_name to the query object.

Revision ID: 72428d1ea401
Revises: 0a6f12f60c73
Create Date: 2020-02-20 08:52:22.877902

"""
revision = '72428d1ea401'
down_revision = '0a6f12f60c73'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('query', sa.Column('tmp_schema_name', sa.String(length=256), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    try:
        op.drop_column('query', 'tmp_schema_name')
    except Exception:
        pass