"""add_qa_document_language

Revision ID: 2c8af9671032
Revises: 8d2d099ceb74
Create Date: 2023-08-01 18:57:27.294973

"""
from alembic import op
import sqlalchemy as sa
revision = '2c8af9671032'
down_revision = '5022897aaceb'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('doc_language', sa.String(length=255), nullable=True))

def downgrade():
    if False:
        return 10
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_column('doc_language')