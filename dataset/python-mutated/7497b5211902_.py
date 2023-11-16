"""Make EnvironmentImage.digest nullable

Revision ID: 7497b5211902
Revises: 6ca72589ff64
Create Date: 2022-08-23 13:37:12.065720

"""
import sqlalchemy as sa
from alembic import op
revision = '7497b5211902'
down_revision = '6ca72589ff64'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.alter_column('environment_images', 'digest', existing_type=sa.VARCHAR(length=71), nullable=True, server_default=None)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.alter_column('environment_images', 'digest', existing_type=sa.VARCHAR(length=71), nullable=False, server_default=None)