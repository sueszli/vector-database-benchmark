"""Adding ephemeral change datetime column to itemrevision.

Revision ID: ae5c0a6aebb3
Revises: 61a6fd4b4500
Create Date: 2016-05-16 17:47:00.946859

"""
revision = 'ae5c0a6aebb3'
down_revision = '61a6fd4b4500'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        return 10
    op.add_column('itemrevision', sa.Column('date_last_ephemeral_change', sa.DateTime(), nullable=True))
    op.create_index('ix_itemrevision_date_last_ephemeral_change', 'itemrevision', ['date_last_ephemeral_change'], unique=False)

def downgrade():
    if False:
        return 10
    op.drop_index('ix_itemrevision_date_last_ephemeral_change', table_name='itemrevision')
    op.drop_column('itemrevision', 'date_last_ephemeral_change')