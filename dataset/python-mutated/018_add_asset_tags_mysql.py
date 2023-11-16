"""add asset tags.

Revision ID: 3ca619834060
Revises: ba4050312958
Create Date: 2021-03-15 15:08:10.277993

"""
from dagster._core.storage.migration.utils import add_asset_materialization_columns
revision = '3ca619834060'
down_revision = 'ba4050312958'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    add_asset_materialization_columns()

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass