"""add versioned instance config

Revision ID: 523fff3f969c
Revises: 3da3fcab826a
Create Date: 2019-11-02 23:06:12.161868

"""
import sqlalchemy as sa
from alembic import op
revision = '523fff3f969c'
down_revision = '3da3fcab826a'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.create_table('instance_config', sa.Column('version', sa.Integer(), nullable=False), sa.Column('valid_until', sa.DateTime(), nullable=True), sa.Column('allow_document_uploads', sa.Boolean(), nullable=True), sa.PrimaryKeyConstraint('version'), sa.UniqueConstraint('valid_until'))
    conn = op.get_bind()
    conn.execute('INSERT INTO instance_config (allow_document_uploads) VALUES (1)')

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_table('instance_config')