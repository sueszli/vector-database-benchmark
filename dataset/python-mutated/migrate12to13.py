import os
import sqlite3

def do_migration(conf):
    if False:
        i = 10
        return i + 15
    db_path = os.path.join(conf.data_dir, 'lbrynet.sqlite')
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    current_columns = []
    for col_info in cursor.execute("pragma table_info('file');").fetchall():
        current_columns.append(col_info[1])
    if 'bt_infohash' in current_columns:
        connection.close()
        print('already migrated')
        return
    cursor.executescript('\n        pragma foreign_keys=off;\n\n        create table if not exists torrent (\n            bt_infohash char(20) not null primary key,\n            tracker text,\n            length integer not null,\n            name text not null\n        );\n\n        create table if not exists torrent_node ( -- BEP-0005\n            bt_infohash char(20) not null references torrent,\n            host text not null,\n            port integer not null\n        );\n\n        create table if not exists torrent_tracker ( -- BEP-0012\n            bt_infohash char(20) not null references torrent,\n            tracker text not null\n        );\n\n        create table if not exists torrent_http_seed ( -- BEP-0017\n            bt_infohash char(20) not null references torrent,\n            http_seed text not null\n        );\n\n        create table if not exists new_file (\n            stream_hash char(96) references stream,\n            bt_infohash char(20) references torrent,\n            file_name text,\n            download_directory text,\n            blob_data_rate real not null,\n            status text not null,\n            saved_file integer not null,\n            content_fee text,\n            added_on integer not null\n        );\n\n        create table if not exists new_content_claim (\n            stream_hash char(96) references stream,\n            bt_infohash char(20) references torrent,\n            claim_outpoint text unique not null references claim\n        );\n\n        insert into new_file (stream_hash, bt_infohash, file_name, download_directory, blob_data_rate, status,\n            saved_file, content_fee, added_on) select\n                stream_hash, NULL, file_name, download_directory, blob_data_rate, status, saved_file, content_fee,\n                added_on\n            from file;\n\n        insert or ignore into new_content_claim (stream_hash, bt_infohash, claim_outpoint)\n            select stream_hash, NULL, claim_outpoint from content_claim;\n\n        drop table file;\n        drop table content_claim;\n        alter table new_file rename to file;\n        alter table new_content_claim rename to content_claim;\n\n        pragma foreign_keys=on;\n    ')
    connection.commit()
    connection.close()