"""empty message

Revision ID: 410e08270de4
Revises: 6bbe0b4a8c4a
Create Date: 2022-04-21 13:40:06.120761

"""
import sqlalchemy as sa
from alembic import op
revision = '410e08270de4'
down_revision = '6bbe0b4a8c4a'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('event_types', sa.Column('name', sa.String(length=50), nullable=False), sa.PrimaryKeyConstraint('name', name=op.f('pk_event_types')))
    op.execute("\n        INSERT INTO event_types (name) values\n        ('project:one-off-job:created'),\n        ('project:one-off-job:started'),\n        ('project:one-off-job:deleted'),\n        ('project:one-off-job:cancelled'),\n        ('project:one-off-job:failed'),\n        ('project:one-off-job:succeeded')\n        ;\n        ")

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('event_types')