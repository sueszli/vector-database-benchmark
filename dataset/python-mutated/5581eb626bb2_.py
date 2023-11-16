"""empty message

Revision ID: 5581eb626bb2
Revises: 69aa6e1d358a
Create Date: 2022-12-29 13:55:20.140761

"""
import sqlalchemy as sa
from alembic import op
revision = '5581eb626bb2'
down_revision = '69aa6e1d358a'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.drop_table('background_tasks')

def downgrade():
    if False:
        return 10
    op.create_table('background_tasks', sa.Column('uuid', sa.VARCHAR(length=36), autoincrement=False, nullable=False), sa.Column('task_type', sa.VARCHAR(length=50), autoincrement=False, nullable=True), sa.Column('status', sa.VARCHAR(length=15), autoincrement=False, nullable=False), sa.Column('code', sa.VARCHAR(length=15), autoincrement=False, nullable=True), sa.Column('result', sa.VARCHAR(), autoincrement=False, nullable=True), sa.PrimaryKeyConstraint('uuid', name='pk_background_tasks'), sa.UniqueConstraint('uuid', name='uq_background_tasks_uuid'))