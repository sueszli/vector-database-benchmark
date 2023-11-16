"""package activity migration check

Revision ID: 23c92480926e
Revises: 3537d5420e0e
Create Date: 2019-05-09 13:39:17.486611

"""
from __future__ import print_function
from alembic import op
from ckan.migration.migrate_package_activity import num_unmigrated
revision = u'23c92480926e'
down_revision = u'3537d5420e0e'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    conn = op.get_bind()
    num_unmigrated_dataset_activities = num_unmigrated(conn)
    if num_unmigrated_dataset_activities:
        print(u"\nNOTE:\nYou have {num_unmigrated} unmigrated package activities.\n\nOnce your CKAN upgrade is complete and CKAN server is running again, you\nshould run the package activity migration, so that the Activity Stream can\ndisplay the detailed history of datasets:\n\n  python ckan/migration/migrate_package_activity.py -c /etc/ckan/production.ini\n\nOnce you've done that, the detailed history is visible in Activity Stream\nto *admins only*. However you are encouraged to make it available to the\npublic, by setting this in production.ini:\n\n  ckan.auth.public_activity_stream_detail = true\n\nMore information about all of this is here:\nhttps://github.com/ckan/ckan/wiki/Migrate-package-activity\n            ".format(num_unmigrated=num_unmigrated_dataset_activities))
    else:
        are_any_datasets = bool(conn.execute(u'SELECT id FROM PACKAGE LIMIT 1').rowcount)
        if are_any_datasets:
            print(u'You have no unmigrated package activities - you do not need to run migrate_package_activity.py.')

def downgrade():
    if False:
        print('Hello World!')
    pass