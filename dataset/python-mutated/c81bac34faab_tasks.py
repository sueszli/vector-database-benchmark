"""tasks

Revision ID: c81bac34faab
Revises: f7ac3d27bb1d
Create Date: 2017-11-23 10:56:49.599779

"""
from alembic import op
import sqlalchemy as sa
revision = 'c81bac34faab'
down_revision = 'f7ac3d27bb1d'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('task', sa.Column('id', sa.String(length=36), nullable=False), sa.Column('name', sa.String(length=128), nullable=True), sa.Column('description', sa.String(length=128), nullable=True), sa.Column('user_id', sa.Integer(), nullable=True), sa.Column('complete', sa.Boolean(), nullable=True), sa.ForeignKeyConstraint(['user_id'], ['user.id']), sa.PrimaryKeyConstraint('id'))
    op.create_index(op.f('ix_task_name'), 'task', ['name'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index(op.f('ix_task_name'), table_name='task')
    op.drop_table('task')