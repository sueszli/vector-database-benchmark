"""update_dataset_model_field_null_available

Revision ID: 4bcffcd64aa4
Revises: 853f9b9cd3b6
Create Date: 2023-08-28 20:58:50.077056

"""
from alembic import op
import sqlalchemy as sa
revision = '4bcffcd64aa4'
down_revision = '853f9b9cd3b6'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.alter_column('embedding_model', existing_type=sa.VARCHAR(length=255), nullable=True, existing_server_default=sa.text("'text-embedding-ada-002'::character varying"))
        batch_op.alter_column('embedding_model_provider', existing_type=sa.VARCHAR(length=255), nullable=True, existing_server_default=sa.text("'openai'::character varying"))

def downgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.alter_column('embedding_model_provider', existing_type=sa.VARCHAR(length=255), nullable=False, existing_server_default=sa.text("'openai'::character varying"))
        batch_op.alter_column('embedding_model', existing_type=sa.VARCHAR(length=255), nullable=False, existing_server_default=sa.text("'text-embedding-ada-002'::character varying"))