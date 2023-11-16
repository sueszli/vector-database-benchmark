"""Resource Modified

Revision ID: 2f97c068fab9
Revises: a91808a89623
Create Date: 2023-06-02 13:13:21.670935

"""
from alembic import op
import sqlalchemy as sa
revision = '2f97c068fab9'
down_revision = 'a91808a89623'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.add_column('resources', sa.Column('agent_id', sa.Integer(), nullable=True))
    op.drop_column('resources', 'project_id')

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.add_column('resources', sa.Column('project_id', sa.INTEGER(), autoincrement=False, nullable=True))
    op.drop_column('resources', 'agent_id')