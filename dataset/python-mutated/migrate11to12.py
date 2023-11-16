import sqlite3
import os
import time

def do_migration(conf):
    if False:
        print('Hello World!')
    db_path = os.path.join(conf.data_dir, 'lbrynet.sqlite')
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    current_columns = []
    for col_info in cursor.execute("pragma table_info('file');").fetchall():
        current_columns.append(col_info[1])
    if 'added_on' in current_columns:
        connection.close()
        print('already migrated')
        return
    cursor.execute('pragma foreign_keys=off')
    cursor.execute('drop table if exists new_file')
    cursor.execute('\n        create table if not exists new_file (\n            stream_hash         text    not null    primary key     references stream,\n            file_name           text,\n            download_directory  text,\n            blob_data_rate      text    not null,\n            status              text    not null,\n            saved_file          integer not null,\n            content_fee         text,\n            added_on            integer not null\n        );\n\n\n    ')
    select = 'select * from file'
    for (stream_hash, file_name, download_dir, blob_rate, status, saved_file, fee) in cursor.execute(select).fetchall():
        added_on = int(time.time())
        cursor.execute('insert into new_file values (?, ?, ?, ?, ?, ?, ?, ?)', (stream_hash, file_name, download_dir, blob_rate, status, saved_file, fee, added_on))
    cursor.execute('drop table file')
    cursor.execute('alter table new_file rename to file')
    cursor.execute('pragma foreign_key_check;')
    connection.commit()
    connection.execute('pragma foreign_keys=on;')
    connection.close()