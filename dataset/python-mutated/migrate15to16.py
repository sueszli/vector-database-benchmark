import os
import sqlite3

def do_migration(conf):
    if False:
        for i in range(10):
            print('nop')
    db_path = os.path.join(conf.data_dir, 'lbrynet.sqlite')
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.executescript('\n        update blob set should_announce=0\n        where should_announce=1 and \n        blob.blob_hash in (select stream_blob.blob_hash from stream_blob where position=0);\n    ')
    connection.commit()
    connection.close()