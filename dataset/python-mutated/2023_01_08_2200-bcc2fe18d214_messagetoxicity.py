"""MessageToxicity

Revision ID: bcc2fe18d214
Revises: 20cd871f4ec7
Create Date: 2023-01-08 22:00:43.297719

"""
import sqlalchemy as sa
import sqlmodel
from alembic import op
from sqlalchemy.dialects import postgresql
revision = 'bcc2fe18d214'
down_revision = '846cc08ac79f'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    op.create_table('message_toxicity', sa.Column('message_id', postgresql.UUID(as_uuid=True), nullable=False), sa.Column('toxicity', sa.Float(), nullable=True), sa.Column('created_date', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False), sa.Column('model', sqlmodel.sql.sqltypes.AutoString(length=256), nullable=False), sa.ForeignKeyConstraint(['message_id'], ['message.id']), sa.PrimaryKeyConstraint('message_id', 'model'))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('message_toxicity')