import os
import sqlite3

def do_migration(conf):
    if False:
        print('Hello World!')
    db_path = os.path.join(conf.data_dir, 'lbrynet.sqlite')
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.executescript('\n        alter table blob add column added_on integer not null default 0;\n        alter table blob add column is_mine integer not null default 1;\n    ')
    connection.commit()
    connection.close()