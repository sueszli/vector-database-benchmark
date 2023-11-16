"""Index deployment.created

Revision ID: ad4b1b4d1e9d
Revises: 22b7cb02e593
Create Date: 2022-10-14 17:26:12.326496

"""
from alembic import op
revision = 'ad4b1b4d1e9d'
down_revision = '22b7cb02e593'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    with op.batch_alter_table('deployment', schema=None) as batch_op:
        batch_op.create_index('ix_deployment__created', ['created'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('deployment', schema=None) as batch_op:
        batch_op.drop_index('ix_deployment__created')