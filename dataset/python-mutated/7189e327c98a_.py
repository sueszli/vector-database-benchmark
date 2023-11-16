"""Add environment_image.digest column

Revision ID: 7189e327c98a
Revises: 11462a4539f6
Create Date: 2022-04-12 10:07:35.329014

"""
import sqlalchemy as sa
from alembic import op
revision = '7189e327c98a'
down_revision = '11462a4539f6'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('environment_images', sa.Column('digest', sa.String(length=71), server_default='Undefined', nullable=False))
    op.create_index(op.f('ix_environment_images_digest'), 'environment_images', ['digest'], unique=False)

def downgrade():
    if False:
        print('Hello World!')
    op.drop_index(op.f('ix_environment_images_digest'), table_name='environment_images')
    op.drop_column('environment_images', 'digest')