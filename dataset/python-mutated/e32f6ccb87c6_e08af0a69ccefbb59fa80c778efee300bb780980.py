"""e08af0a69ccefbb59fa80c778efee300bb780980

Revision ID: e32f6ccb87c6
Revises: a45f4dfde53b
Create Date: 2023-06-06 19:58:33.103819

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision = 'e32f6ccb87c6'
down_revision = '614f77cecc48'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_table('data_source_bindings', sa.Column('id', postgresql.UUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False), sa.Column('tenant_id', postgresql.UUID(), nullable=False), sa.Column('access_token', sa.String(length=255), nullable=False), sa.Column('provider', sa.String(length=255), nullable=False), sa.Column('source_info', postgresql.JSONB(astext_type=sa.Text()), nullable=False), sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP(0)'), nullable=False), sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP(0)'), nullable=False), sa.Column('disabled', sa.Boolean(), server_default=sa.text('false'), nullable=True), sa.PrimaryKeyConstraint('id', name='source_binding_pkey'))
    with op.batch_alter_table('data_source_bindings', schema=None) as batch_op:
        batch_op.create_index('source_binding_tenant_id_idx', ['tenant_id'], unique=False)
        batch_op.create_index('source_info_idx', ['source_info'], unique=False, postgresql_using='gin')

def downgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('data_source_bindings', schema=None) as batch_op:
        batch_op.drop_index('source_info_idx', postgresql_using='gin')
        batch_op.drop_index('source_binding_tenant_id_idx')
    op.drop_table('data_source_bindings')