"""create submission uuid column

Revision ID: fccf57ceef02
Revises: 3d91d6948753
Create Date: 2018-07-12 00:06:20.891213

"""
import uuid
import sqlalchemy as sa
from alembic import op
revision = 'fccf57ceef02'
down_revision = '3d91d6948753'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    conn = op.get_bind()
    conn.execute('PRAGMA legacy_alter_table=ON')
    op.rename_table('submissions', 'submissions_tmp')
    op.add_column('submissions_tmp', sa.Column('uuid', sa.String(length=36)))
    submissions = conn.execute(sa.text('SELECT * FROM submissions_tmp')).fetchall()
    for submission in submissions:
        conn.execute(sa.text('UPDATE submissions_tmp SET uuid=:submission_uuid WHERE\n                       id=:id').bindparams(submission_uuid=str(uuid.uuid4()), id=submission.id))
    op.create_table('submissions', sa.Column('id', sa.Integer(), nullable=False), sa.Column('uuid', sa.String(length=36), nullable=False), sa.Column('source_id', sa.Integer(), nullable=True), sa.Column('filename', sa.String(length=255), nullable=False), sa.Column('size', sa.Integer(), nullable=False), sa.Column('downloaded', sa.Boolean(), nullable=True), sa.ForeignKeyConstraint(['source_id'], ['sources.id']), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('uuid'))
    conn.execute('\n        INSERT INTO submissions\n        SELECT id, uuid, source_id, filename, size, downloaded\n        FROM submissions_tmp\n    ')
    op.drop_table('submissions_tmp')

def downgrade() -> None:
    if False:
        while True:
            i = 10
    with op.batch_alter_table('submissions', schema=None) as batch_op:
        batch_op.drop_column('uuid')