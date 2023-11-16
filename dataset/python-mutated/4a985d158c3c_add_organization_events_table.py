"""
add_organization_events_table

Revision ID: 4a985d158c3c
Revises: 614a7fcb40ed
Create Date: 2022-04-14 02:25:50.805348
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '4a985d158c3c'
down_revision = '614a7fcb40ed'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_table('organization_events', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False), sa.Column('source_id', postgresql.UUID(as_uuid=True), nullable=False), sa.Column('tag', sa.String(), nullable=False), sa.Column('time', sa.DateTime(), server_default=sa.text('now()'), nullable=False), sa.Column('ip_address', sa.String(), nullable=False), sa.Column('additional', postgresql.JSONB(astext_type=sa.Text()), nullable=True), sa.ForeignKeyConstraint(['source_id'], ['organizations.id'], initially='DEFERRED', deferrable=True), sa.PrimaryKeyConstraint('id'))
    op.create_index(op.f('ix_organization_events_source_id'), 'organization_events', ['source_id'], unique=False)

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_index(op.f('ix_organization_events_source_id'), table_name='organization_events')
    op.drop_table('organization_events')