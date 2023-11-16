"""Create source UUID column

Revision ID: 3d91d6948753
Revises: faac8092c123
Create Date: 2018-07-09 22:39:05.088008

"""
import uuid
import sqlalchemy as sa
from alembic import op
revision = '3d91d6948753'
down_revision = 'faac8092c123'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    conn = op.get_bind()
    conn.execute('PRAGMA legacy_alter_table=ON')
    op.rename_table('sources', 'sources_tmp')
    op.add_column('sources_tmp', sa.Column('uuid', sa.String(length=36)))
    sources = conn.execute(sa.text('SELECT * FROM sources_tmp')).fetchall()
    for source in sources:
        conn.execute(sa.text('UPDATE sources_tmp SET uuid=:source_uuid WHERE\n                       id=:id').bindparams(source_uuid=str(uuid.uuid4()), id=source.id))
    op.create_table('sources', sa.Column('id', sa.Integer(), nullable=False), sa.Column('uuid', sa.String(length=36), nullable=False), sa.Column('filesystem_id', sa.String(length=96), nullable=True), sa.Column('journalist_designation', sa.String(length=255), nullable=False), sa.Column('flagged', sa.Boolean(), nullable=True), sa.Column('last_updated', sa.DateTime(), nullable=True), sa.Column('pending', sa.Boolean(), nullable=True), sa.Column('interaction_count', sa.Integer(), nullable=False), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('uuid'), sa.UniqueConstraint('filesystem_id'))
    conn.execute('\n        INSERT INTO sources\n        SELECT id, uuid, filesystem_id, journalist_designation, flagged,\n               last_updated, pending, interaction_count\n        FROM sources_tmp\n    ')
    op.drop_table('sources_tmp')

def downgrade() -> None:
    if False:
        return 10
    with op.batch_alter_table('sources', schema=None) as batch_op:
        batch_op.drop_column('uuid')