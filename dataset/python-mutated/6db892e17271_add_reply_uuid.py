"""add reply UUID

Revision ID: 6db892e17271
Revises: e0a525cbab83
Create Date: 2018-08-06 20:31:50.035066

"""
import uuid
import sqlalchemy as sa
from alembic import op
revision = '6db892e17271'
down_revision = 'e0a525cbab83'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    conn = op.get_bind()
    conn.execute('PRAGMA legacy_alter_table=ON')
    op.rename_table('replies', 'replies_tmp')
    op.add_column('replies_tmp', sa.Column('uuid', sa.String(length=36)))
    replies = conn.execute(sa.text('SELECT * FROM replies_tmp')).fetchall()
    for reply in replies:
        conn.execute(sa.text('UPDATE replies_tmp SET uuid=:reply_uuid WHERE\n                       id=:id').bindparams(reply_uuid=str(uuid.uuid4()), id=reply.id))
    op.create_table('replies', sa.Column('id', sa.Integer(), nullable=False), sa.Column('uuid', sa.String(length=36), nullable=False), sa.Column('journalist_id', sa.Integer(), nullable=True), sa.Column('source_id', sa.Integer(), nullable=True), sa.Column('filename', sa.String(length=255), nullable=False), sa.Column('size', sa.Integer(), nullable=False), sa.Column('deleted_by_source', sa.Boolean(), nullable=False), sa.ForeignKeyConstraint(['journalist_id'], ['journalists.id']), sa.ForeignKeyConstraint(['source_id'], ['sources.id']), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('uuid'))
    conn.execute('\n        INSERT INTO replies\n        SELECT id, uuid, journalist_id, source_id, filename, size,\n            deleted_by_source\n        FROM replies_tmp\n    ')
    op.drop_table('replies_tmp')

def downgrade() -> None:
    if False:
        return 10
    with op.batch_alter_table('replies', schema=None) as batch_op:
        batch_op.drop_column('uuid')