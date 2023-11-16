"""
create table for warehouse administration flags

Revision ID: 7165e957cddc
Revises: 1e2ccd34f539
Create Date: 2018-02-17 18:42:18.209572
"""
import sqlalchemy as sa
from alembic import op
revision = '7165e957cddc'
down_revision = '1e2ccd34f539'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_table('warehouse_admin_flag', sa.Column('id', sa.Text(), nullable=False), sa.Column('description', sa.Text(), nullable=False), sa.Column('enabled', sa.Boolean(), nullable=False), sa.PrimaryKeyConstraint('id'))
    op.execute("\n        INSERT INTO warehouse_admin_flag(id, description, enabled)\n        VALUES (\n            'disallow-new-user-registration',\n            'Disallow ALL new User registrations',\n            FALSE\n        )\n    ")
    op.execute("\n        INSERT INTO warehouse_admin_flag(id, description, enabled)\n        VALUES (\n            'disallow-new-project-registration',\n            'Disallow ALL new Project registrations',\n            FALSE\n        )\n    ")

def downgrade():
    if False:
        print('Hello World!')
    op.drop_table('warehouse_admin_flag')