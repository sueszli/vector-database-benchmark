"""message review_result nullable

Revision ID: 8cd0c34d0c3c
Revises: 165b55de5a94
Create Date: 2023-02-15 17:54:58.029278

"""
import sqlalchemy as sa
from alembic import op
revision = '8cd0c34d0c3c'
down_revision = '165b55de5a94'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    op.alter_column('message', 'review_result', existing_type=sa.BOOLEAN(), nullable=True, server_default=None, existing_server_default=sa.text('false'))

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.alter_column('message', 'review_result', existing_type=sa.BOOLEAN(), nullable=False, server_default=sa.text('false'))