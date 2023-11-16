"""
Add psf_staff boolean flag

Revision ID: 590c513f1c74
Revises: d0c22553b338
Create Date: 2021-06-07 11:49:50.688410
"""
import sqlalchemy as sa
from alembic import op
revision = '590c513f1c74'
down_revision = 'd0c22553b338'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('users', sa.Column('is_psf_staff', sa.Boolean(), server_default=sa.sql.false(), nullable=False))

def downgrade():
    if False:
        return 10
    op.drop_column('users', 'is_psf_staff')