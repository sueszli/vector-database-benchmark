"""add-index-to-flow-run-infrastructure_document_id

Revision ID: fa985d474982
Revises: add97ce1937d
Create Date: 2022-07-29 18:17:13.174765

"""
from alembic import op
revision = 'fa985d474982'
down_revision = 'add97ce1937d'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.get_context().autocommit_block():
        op.create_index(op.f('ix_flow_run__infrastructure_document_id'), 'flow_run', ['infrastructure_document_id'], unique=False, postgresql_concurrently=True)

def downgrade():
    if False:
        i = 10
        return i + 15
    with op.get_context().autocommit_block():
        op.drop_index(op.f('ix_flow_run__infrastructure_document_id'), table_name='flow_run', postgresql_concurrently=True)