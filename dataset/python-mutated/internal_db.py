import textwrap
from datasette.utils import table_column_details

async def init_internal_db(db):
    create_tables_sql = textwrap.dedent('\n    CREATE TABLE IF NOT EXISTS catalog_databases (\n        database_name TEXT PRIMARY KEY,\n        path TEXT,\n        is_memory INTEGER,\n        schema_version INTEGER\n    );\n    CREATE TABLE IF NOT EXISTS catalog_tables (\n        database_name TEXT,\n        table_name TEXT,\n        rootpage INTEGER,\n        sql TEXT,\n        PRIMARY KEY (database_name, table_name),\n        FOREIGN KEY (database_name) REFERENCES databases(database_name)\n    );\n    CREATE TABLE IF NOT EXISTS catalog_columns (\n        database_name TEXT,\n        table_name TEXT,\n        cid INTEGER,\n        name TEXT,\n        type TEXT,\n        "notnull" INTEGER,\n        default_value TEXT, -- renamed from dflt_value\n        is_pk INTEGER, -- renamed from pk\n        hidden INTEGER,\n        PRIMARY KEY (database_name, table_name, name),\n        FOREIGN KEY (database_name) REFERENCES databases(database_name),\n        FOREIGN KEY (database_name, table_name) REFERENCES tables(database_name, table_name)\n    );\n    CREATE TABLE IF NOT EXISTS catalog_indexes (\n        database_name TEXT,\n        table_name TEXT,\n        seq INTEGER,\n        name TEXT,\n        "unique" INTEGER,\n        origin TEXT,\n        partial INTEGER,\n        PRIMARY KEY (database_name, table_name, name),\n        FOREIGN KEY (database_name) REFERENCES databases(database_name),\n        FOREIGN KEY (database_name, table_name) REFERENCES tables(database_name, table_name)\n    );\n    CREATE TABLE IF NOT EXISTS catalog_foreign_keys (\n        database_name TEXT,\n        table_name TEXT,\n        id INTEGER,\n        seq INTEGER,\n        "table" TEXT,\n        "from" TEXT,\n        "to" TEXT,\n        on_update TEXT,\n        on_delete TEXT,\n        match TEXT,\n        PRIMARY KEY (database_name, table_name, id, seq),\n        FOREIGN KEY (database_name) REFERENCES databases(database_name),\n        FOREIGN KEY (database_name, table_name) REFERENCES tables(database_name, table_name)\n    );\n    ').strip()
    await db.execute_write_script(create_tables_sql)

async def populate_schema_tables(internal_db, db):
    database_name = db.name

    def delete_everything(conn):
        if False:
            while True:
                i = 10
        conn.execute('DELETE FROM catalog_tables WHERE database_name = ?', [database_name])
        conn.execute('DELETE FROM catalog_columns WHERE database_name = ?', [database_name])
        conn.execute('DELETE FROM catalog_foreign_keys WHERE database_name = ?', [database_name])
        conn.execute('DELETE FROM catalog_indexes WHERE database_name = ?', [database_name])
    await internal_db.execute_write_fn(delete_everything)
    tables = (await db.execute("select * from sqlite_master WHERE type = 'table'")).rows

    def collect_info(conn):
        if False:
            for i in range(10):
                print('nop')
        tables_to_insert = []
        columns_to_insert = []
        foreign_keys_to_insert = []
        indexes_to_insert = []
        for table in tables:
            table_name = table['name']
            tables_to_insert.append((database_name, table_name, table['rootpage'], table['sql']))
            columns = table_column_details(conn, table_name)
            columns_to_insert.extend(({**{'database_name': database_name, 'table_name': table_name}, **column._asdict()} for column in columns))
            foreign_keys = conn.execute(f'PRAGMA foreign_key_list([{table_name}])').fetchall()
            foreign_keys_to_insert.extend(({**{'database_name': database_name, 'table_name': table_name}, **dict(foreign_key)} for foreign_key in foreign_keys))
            indexes = conn.execute(f'PRAGMA index_list([{table_name}])').fetchall()
            indexes_to_insert.extend(({**{'database_name': database_name, 'table_name': table_name}, **dict(index)} for index in indexes))
        return (tables_to_insert, columns_to_insert, foreign_keys_to_insert, indexes_to_insert)
    (tables_to_insert, columns_to_insert, foreign_keys_to_insert, indexes_to_insert) = await db.execute_fn(collect_info)
    await internal_db.execute_write_many('\n        INSERT INTO catalog_tables (database_name, table_name, rootpage, sql)\n        values (?, ?, ?, ?)\n    ', tables_to_insert)
    await internal_db.execute_write_many('\n        INSERT INTO catalog_columns (\n            database_name, table_name, cid, name, type, "notnull", default_value, is_pk, hidden\n        ) VALUES (\n            :database_name, :table_name, :cid, :name, :type, :notnull, :default_value, :is_pk, :hidden\n        )\n    ', columns_to_insert)
    await internal_db.execute_write_many('\n        INSERT INTO catalog_foreign_keys (\n            database_name, table_name, "id", seq, "table", "from", "to", on_update, on_delete, match\n        ) VALUES (\n            :database_name, :table_name, :id, :seq, :table, :from, :to, :on_update, :on_delete, :match\n        )\n    ', foreign_keys_to_insert)
    await internal_db.execute_write_many('\n        INSERT INTO catalog_indexes (\n            database_name, table_name, seq, name, "unique", origin, partial\n        ) VALUES (\n            :database_name, :table_name, :seq, :name, :unique, :origin, :partial\n        )\n    ', indexes_to_insert)