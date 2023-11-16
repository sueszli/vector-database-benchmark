import sqlite3
import os

def do_migration(conf):
    if False:
        print('Hello World!')
    db_path = os.path.join(conf.data_dir, 'lbrynet.sqlite')
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.executescript('\n        create table reflected_stream (\n            sd_hash text not null,\n            reflector_address text not null,\n            timestamp integer,\n            primary key (sd_hash, reflector_address)\n        );\n        ')
    connection.commit()
    connection.close()