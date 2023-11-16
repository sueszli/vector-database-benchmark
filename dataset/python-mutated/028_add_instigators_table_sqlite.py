"""add instigators table.

Revision ID: c892b3fe0a9f
Revises: 16e3115a602a
Create Date: 2022-03-18 16:16:21.007430

"""
from dagster._core.storage.migration.utils import create_instigators_table
revision = 'c892b3fe0a9f'
down_revision = '16e3115a602a'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    create_instigators_table()

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass