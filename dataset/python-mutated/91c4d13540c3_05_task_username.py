"""task_username

Revision ID: 91c4d13540c3
Revises: 9532a372b5aa
Create Date: 2018-12-25 15:19:15.045063

"""
from alembic import op
import sqlalchemy as sa
revision = '91c4d13540c3'
down_revision = '9532a372b5aa'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('tasks', sa.Column('user_name', sa.String(100), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('tasks', 'user_name')