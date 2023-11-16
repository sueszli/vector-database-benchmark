"""Add safe_content to message

Revision ID: ea19bbc743f9
Revises: 401eef162771
Create Date: 2023-04-14 22:37:41.373382

"""
import sqlalchemy as sa
import sqlmodel
from alembic import op
revision = 'ea19bbc743f9'
down_revision = '401eef162771'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.add_column('message', sa.Column('safe_content', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column('message', sa.Column('safety_level', sa.Integer(), nullable=True))
    op.add_column('message', sa.Column('safety_label', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column('message', sa.Column('safety_rots', sqlmodel.sql.sqltypes.AutoString(), nullable=True))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('message', 'safe_content')
    op.drop_column('message', 'safety_level')
    op.drop_column('message', 'safety_label')
    op.drop_column('message', 'safety_rots')