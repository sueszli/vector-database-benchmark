"""delete orphaned submissions and replies

Ref: https://github.com/freedomofpress/securedrop/issues/1189

Revision ID: 3da3fcab826a
Revises: 60f41bb14d98
Create Date: 2018-11-25 19:40:25.873292

"""
import sqlalchemy as sa
from alembic import op
from journalist_app import create_app
from sdconfig import SecureDropConfig
from store import NoFileFoundException, Storage, TooManyFilesException
revision = '3da3fcab826a'
down_revision = '60f41bb14d98'
branch_labels = None
depends_on = None

def raw_sql_grab_orphaned_objects(table_name: str) -> str:
    if False:
        print('Hello World!')
    "Objects that have a source ID that doesn't exist in the\n    sources table OR a NULL source ID should be deleted."
    return 'SELECT id, filename, source_id FROM {table} WHERE source_id NOT IN (SELECT id FROM sources) UNION SELECT id, filename, source_id FROM {table} WHERE source_id IS NULL'.format(table=table_name)

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    try:
        config = SecureDropConfig.get_current()
    except ModuleNotFoundError:
        return
    conn = op.get_bind()
    submissions = conn.execute(sa.text(raw_sql_grab_orphaned_objects('submissions'))).fetchall()
    replies = conn.execute(sa.text(raw_sql_grab_orphaned_objects('replies'))).fetchall()
    app = create_app(config)
    with app.app_context():
        for submission in submissions:
            try:
                conn.execute(sa.text('\n                    DELETE FROM submissions\n                    WHERE id=:id\n                ').bindparams(id=submission.id))
                path = Storage.get_default().path_without_filesystem_id(submission.filename)
                Storage.get_default().move_to_shredder(path)
            except NoFileFoundException:
                conn.execute(sa.text('\n                    DELETE FROM submissions\n                    WHERE id=:id\n                ').bindparams(id=submission.id))
            except TooManyFilesException:
                pass
        for reply in replies:
            try:
                conn.execute(sa.text('\n                        DELETE FROM replies\n                        WHERE id=:id\n                    ').bindparams(id=reply.id))
                path = Storage.get_default().path_without_filesystem_id(reply.filename)
                Storage.get_default().move_to_shredder(path)
            except NoFileFoundException:
                conn.execute(sa.text('\n                        DELETE FROM replies\n                        WHERE id=:id\n                    ').bindparams(id=reply.id))
            except TooManyFilesException:
                pass

def downgrade() -> None:
    if False:
        while True:
            i = 10
    pass