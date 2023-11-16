"""Removes DebugPrintNotification block type

Revision ID: e905fd199258
Revises: 4cdc2ba709a4
Create Date: 2022-07-07 11:28:09.792699

"""
from alembic import op
revision = 'e905fd199258'
down_revision = '4cdc2ba709a4'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.execute("\n        DELETE FROM block_type WHERE name = 'Debug Print Notification'\n        ")

def downgrade():
    if False:
        while True:
            i = 10
    pass