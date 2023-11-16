"""add column to track source deletion of replies

Revision ID: e0a525cbab83
Revises: 2d0ce3ee5bdc
Create Date: 2018-08-02 00:07:59.242510

"""
import sqlalchemy as sa
from alembic import op
revision = 'e0a525cbab83'
down_revision = '2d0ce3ee5bdc'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    conn = op.get_bind()
    conn.execute('PRAGMA legacy_alter_table=ON')
    op.rename_table('replies', 'replies_tmp')
    op.add_column('replies_tmp', sa.Column('deleted_by_source', sa.Boolean()))
    replies = conn.execute(sa.text('SELECT * FROM replies_tmp')).fetchall()
    for reply in replies:
        conn.execute(sa.text('UPDATE replies_tmp SET deleted_by_source=0 WHERE\n                       id=:id').bindparams(id=reply.id))
    op.create_table('replies', sa.Column('id', sa.Integer(), nullable=False), sa.Column('journalist_id', sa.Integer(), nullable=True), sa.Column('source_id', sa.Integer(), nullable=True), sa.Column('filename', sa.String(length=255), nullable=False), sa.Column('size', sa.Integer(), nullable=False), sa.Column('deleted_by_source', sa.Boolean(), nullable=False), sa.ForeignKeyConstraint(['journalist_id'], ['journalists.id']), sa.ForeignKeyConstraint(['source_id'], ['sources.id']), sa.PrimaryKeyConstraint('id'))
    conn.execute('\n        INSERT INTO replies\n        SELECT id, journalist_id, source_id, filename, size, deleted_by_source\n        FROM replies_tmp\n    ')
    op.drop_table('replies_tmp')

def downgrade() -> None:
    if False:
        print('Hello World!')
    with op.batch_alter_table('replies', schema=None) as batch_op:
        batch_op.drop_column('deleted_by_source')