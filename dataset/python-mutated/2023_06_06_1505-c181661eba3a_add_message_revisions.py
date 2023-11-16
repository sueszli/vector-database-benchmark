"""add_message_revisions

Revision ID: c181661eba3a
Revises: 1b6e3ae16e9d
Create Date: 2023-06-06 15:05:58.079120

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = 'c181661eba3a'
down_revision = '1b6e3ae16e9d'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.create_table('message_revision', sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False), sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True), sa.Column('message_id', postgresql.UUID(as_uuid=True), nullable=False), sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True), sa.Column('created_date', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True), sa.ForeignKeyConstraint(['message_id'], ['message.id']), sa.ForeignKeyConstraint(['user_id'], ['user.id']), sa.PrimaryKeyConstraint('id'))
    op.create_index(op.f('ix_message_revision_message_id'), 'message_revision', ['message_id'], unique=False)
    op.add_column('message', sa.Column('edited', sa.Boolean(), server_default=sa.text('false'), nullable=False))

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_column('message', 'edited')
    op.drop_index(op.f('ix_message_revision_message_id'), table_name='message_revision')
    op.drop_table('message_revision')