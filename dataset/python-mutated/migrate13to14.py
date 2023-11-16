import os
import sqlite3

def do_migration(conf):
    if False:
        i = 10
        return i + 15
    db_path = os.path.join(conf.data_dir, 'lbrynet.sqlite')
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.executescript('\n        create table if not exists peer (\n            node_id char(96) not null primary key,\n            address text not null,\n            udp_port integer not null,\n            tcp_port integer,\n            unique (address, udp_port)\n        );\n    ')
    connection.commit()
    connection.close()