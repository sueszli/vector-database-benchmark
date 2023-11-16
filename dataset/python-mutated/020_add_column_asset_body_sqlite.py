"""add column asset body.

Revision ID: 3b529ad30626
Revises: 7cba9eeaaf1d
Create Date: 2021-03-17 16:38:09.418235

"""
from dagster._core.storage.migration.utils import add_asset_details_column
revision = '3b529ad30626'
down_revision = '7cba9eeaaf1d'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    add_asset_details_column()

def downgrade():
    if False:
        return 10
    pass