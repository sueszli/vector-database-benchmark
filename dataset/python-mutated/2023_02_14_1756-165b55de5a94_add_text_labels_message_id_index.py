"""add text_labels message_id index

Revision ID: 165b55de5a94
Revises: ba40d055714a
Create Date: 2023-02-14 17:56:48.263684

"""
from alembic import op
revision = '165b55de5a94'
down_revision = 'ba40d055714a'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.create_index(op.f('ix_text_labels_message_id'), 'text_labels', ['message_id'], unique=False)

def downgrade() -> None:
    if False:
        print('Hello World!')
    op.drop_index(op.f('ix_text_labels_message_id'), table_name='text_labels')