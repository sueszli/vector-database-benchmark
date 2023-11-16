import os
import sqlite3
from core.genToken import genToken, genQRCode

def initDB(DATABASE):
    if False:
        print('Hello World!')
    if not os.path.exists(DATABASE):
        conn = sqlite3.connect(DATABASE)
        cur = conn.cursor()
        create_table_sql = ' CREATE TABLE IF NOT EXISTS creds (\n                                            id integer PRIMARY KEY,\n                                            url text NOT NULL,\n                                            jdoc text,\n                                            pdate numeric,\n                                            browser text,\n                                            bversion text,\n                                            platform text,\n                                            rip text\n                                        ); '
        cur.execute(create_table_sql)
        conn.commit()
        create_table_sql2 = ' CREATE TABLE IF NOT EXISTS socialfish (\n                                            id integer PRIMARY KEY,\n                                            clicks integer,\n                                            attacks integer,\n                                            token text\n                                        ); '
        cur.execute(create_table_sql2)
        conn.commit()
        sql = ' INSERT INTO socialfish(id,clicks,attacks,token)\n                  VALUES(?,?,?,?) '
        i = 1
        c = 0
        a = 0
        t = genToken()
        data = (i, c, a, t)
        cur.execute(sql, data)
        conn.commit()
        create_table_sql3 = ' CREATE TABLE IF NOT EXISTS sfmail (\n                                            id integer PRIMARY KEY,\n                                            email VARCHAR,\n                                            smtp text,\n                                            port text\n                                        ); '
        cur.execute(create_table_sql3)
        conn.commit()
        sql = ' INSERT INTO sfmail(id,email,smtp,port)\n                  VALUES(?,?,?,?) '
        i = 1
        e = ''
        s = ''
        p = ''
        data = (i, e, s, p)
        cur.execute(sql, data)
        conn.commit()
        create_table_sql4 = ' CREATE TABLE IF NOT EXISTS professionals (\n                                            id integer PRIMARY KEY,\n                                            email VARCHAR,\n                                            name text,\n                                            obs text\n                                        ); '
        cur.execute(create_table_sql4)
        conn.commit()
        create_table_sql5 = ' CREATE TABLE IF NOT EXISTS companies (\n                                            id integer PRIMARY KEY,\n                                            email VARCHAR,\n                                            name text,\n                                            phone text,\n                                            address text,\n                                            site text\n                                        ); '
        cur.execute(create_table_sql5)
        conn.commit()
        conn.close()
        genQRCode(t)