"""add asset tags.

Revision ID: 6e1f65d78b92
Revises: b4e0c470acb3
Create Date: 2021-03-04 10:01:01.877875

"""
from dagster._core.storage.migration.utils import add_asset_materialization_columns
revision = '6e1f65d78b92'
down_revision = 'b4e0c470acb3'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    add_asset_materialization_columns()

def downgrade():
    if False:
        print('Hello World!')
    pass