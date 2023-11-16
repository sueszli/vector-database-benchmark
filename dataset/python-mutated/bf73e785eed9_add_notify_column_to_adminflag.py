"""
Add notify column to AdminFlag

Revision ID: bf73e785eed9
Revises: 5dda74213989
Create Date: 2018-03-23 21:20:05.834821
"""
import sqlalchemy as sa
from alembic import op
revision = 'bf73e785eed9'
down_revision = '5dda74213989'

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('warehouse_admin_flag', sa.Column('notify', sa.Boolean(), server_default=sa.text('false'), nullable=False))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('warehouse_admin_flag', 'notify')