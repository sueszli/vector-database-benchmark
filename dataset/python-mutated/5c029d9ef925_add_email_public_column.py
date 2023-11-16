"""
add email.public column

Revision ID: 5c029d9ef925
Revises: e133fc5aa3c1
Create Date: 2020-01-19 22:25:53.901148
"""
import sqlalchemy as sa
from alembic import op
revision = '5c029d9ef925'
down_revision = 'e133fc5aa3c1'

def upgrade():
    if False:
        return 10
    op.add_column('user_emails', sa.Column('public', sa.Boolean(), server_default=sa.text('false'), nullable=False))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('user_emails', 'public')