"""add registration IP

Revision ID: f703f911d4ae
Revises: f69d7fec88d6
Create Date: 2018-07-09 13:04:50.652781

"""
from alembic import op
import sqlalchemy as sa
revision = 'f703f911d4ae'
down_revision = 'f69d7fec88d6'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('users', sa.Column('registration_ip', sa.Binary(), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('users', 'registration_ip')