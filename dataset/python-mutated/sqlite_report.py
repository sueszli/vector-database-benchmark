import sqlite3
from lib.reports.base import SQLBaseReport

class SQLiteReport(SQLBaseReport):

    def connect(self, output_file):
        if False:
            i = 10
            return i + 15
        self.conn = sqlite3.connect(output_file, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def create_table_query(self, table):
        if False:
            print('Hello World!')
        return (f'CREATE TABLE "{table}" (\n            time DATETIME DEFAULT CURRENT_TIMESTAMP,\n            url TEXT,\n            status_code INTEGER,\n            content_length INTEGER,\n            content_type TEXT,\n            redirect TEXT\n        );',)

    def insert_table_query(self, table, values):
        if False:
            return 10
        return (f'INSERT INTO "{table}" (url, status_code, content_length, content_type, redirect)\n                    VALUES\n                    (?, ?, ?, ?, ?)', values)