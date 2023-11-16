"""create conditinal index on user

Revision ID: ff13667243ed
Revises: d111f446733b
Create Date: 2022-06-22 11:38:27.553446

"""
from alembic import op
import sqlalchemy as sa
revision = 'ff13667243ed'
down_revision = 'd111f446733b'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.create_index('idx_only_one_active_email', 'user', ['email', 'state'], unique=True, postgresql_where=sa.text('"user".state=\'active\''))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_index('idx_only_one_active_email')