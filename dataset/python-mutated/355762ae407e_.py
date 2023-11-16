"""Add Setting.requires_restart field

Revision ID: 355762ae407e
Revises: 063e3d096399
Create Date: 2022-05-27 10:05:16.967270

"""
import sqlalchemy as sa
from alembic import op
revision = '355762ae407e'
down_revision = '063e3d096399'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('settings', sa.Column('requires_restart', sa.Boolean(), server_default='False', nullable=False))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('settings', 'requires_restart')