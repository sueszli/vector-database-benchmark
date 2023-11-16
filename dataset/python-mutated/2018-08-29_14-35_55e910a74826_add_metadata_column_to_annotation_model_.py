"""add_metadata_column_to_annotation_model.py

Revision ID: 55e910a74826
Revises: 1a1d627ebd8e
Create Date: 2018-08-29 14:35:20.407743

"""
revision = '55e910a74826'
down_revision = '1a1d627ebd8e'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('annotation', sa.Column('json_metadata', sa.Text(), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('annotation', 'json_metadata')