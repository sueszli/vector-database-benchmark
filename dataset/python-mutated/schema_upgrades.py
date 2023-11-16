__license__ = 'GPL v3'
__copyright__ = '2010, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
import os
from calibre.utils.date import isoformat, DEFAULT_DATE
from polyglot.builtins import itervalues

class SchemaUpgrade:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        while True:
            uv = self.user_version
            meth = getattr(self, 'upgrade_version_%d' % uv, None)
            if meth is None:
                break
            else:
                print('Upgrading database to version %d...' % (uv + 1))
                meth()
                self.user_version = uv + 1

    def upgrade_version_1(self):
        if False:
            print('Hello World!')
        '\n        Normalize indices.\n        '
        self.conn.executescript('        DROP INDEX authors_idx;\n        CREATE INDEX authors_idx ON books (author_sort COLLATE NOCASE, sort COLLATE NOCASE);\n        DROP INDEX series_idx;\n        CREATE INDEX series_idx ON series (name COLLATE NOCASE);\n        CREATE INDEX series_sort_idx ON books (series_index, id);\n        ')

    def upgrade_version_2(self):
        if False:
            for i in range(10):
                print('nop')
        ' Fix Foreign key constraints for deleting from link tables. '
        script = "        DROP TRIGGER IF EXISTS fkc_delete_books_%(ltable)s_link;\n        CREATE TRIGGER fkc_delete_on_%(table)s\n        BEFORE DELETE ON %(table)s\n        BEGIN\n            SELECT CASE\n                WHEN (SELECT COUNT(id) FROM books_%(ltable)s_link WHERE %(ltable_col)s=OLD.id) > 0\n                THEN RAISE(ABORT, 'Foreign key violation: %(table)s is still referenced')\n            END;\n        END;\n        DELETE FROM %(table)s WHERE (SELECT COUNT(id) FROM books_%(ltable)s_link WHERE %(ltable_col)s=%(table)s.id) < 1;\n        "
        self.conn.executescript(script % dict(ltable='authors', table='authors', ltable_col='author'))
        self.conn.executescript(script % dict(ltable='publishers', table='publishers', ltable_col='publisher'))
        self.conn.executescript(script % dict(ltable='tags', table='tags', ltable_col='tag'))
        self.conn.executescript(script % dict(ltable='series', table='series', ltable_col='series'))

    def upgrade_version_3(self):
        if False:
            return 10
        ' Add path to result cache '
        self.conn.executescript('\n        DROP VIEW meta;\n        CREATE VIEW meta AS\n        SELECT id, title,\n               (SELECT concat(name) FROM authors WHERE authors.id IN (SELECT author from books_authors_link WHERE book=books.id)) authors,\n               (SELECT name FROM publishers WHERE publishers.id IN (SELECT publisher from books_publishers_link WHERE book=books.id)) publisher,\n               (SELECT rating FROM ratings WHERE ratings.id IN (SELECT rating from books_ratings_link WHERE book=books.id)) rating,\n               timestamp,\n               (SELECT MAX(uncompressed_size) FROM data WHERE book=books.id) size,\n               (SELECT concat(name) FROM tags WHERE tags.id IN (SELECT tag from books_tags_link WHERE book=books.id)) tags,\n               (SELECT text FROM comments WHERE book=books.id) comments,\n               (SELECT name FROM series WHERE series.id IN (SELECT series FROM books_series_link WHERE book=books.id)) series,\n               series_index,\n               sort,\n               author_sort,\n               (SELECT concat(format) FROM data WHERE data.book=books.id) formats,\n               isbn,\n               path\n        FROM books;\n        ')

    def upgrade_version_4(self):
        if False:
            return 10
        'Rationalize books table'
        self.conn.executescript('\n        BEGIN TRANSACTION;\n        CREATE TEMPORARY TABLE\n        books_backup(id,title,sort,timestamp,series_index,author_sort,isbn,path);\n        INSERT INTO books_backup SELECT id,title,sort,timestamp,series_index,author_sort,isbn,path FROM books;\n        DROP TABLE books;\n        CREATE TABLE books ( id      INTEGER PRIMARY KEY AUTOINCREMENT,\n                             title     TEXT NOT NULL DEFAULT \'Unknown\' COLLATE NOCASE,\n                             sort      TEXT COLLATE NOCASE,\n                             timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                             pubdate   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                             series_index REAL NOT NULL DEFAULT 1.0,\n                             author_sort TEXT COLLATE NOCASE,\n                             isbn TEXT DEFAULT "" COLLATE NOCASE,\n                             lccn TEXT DEFAULT "" COLLATE NOCASE,\n                             path TEXT NOT NULL DEFAULT "",\n                             flags INTEGER NOT NULL DEFAULT 1\n                        );\n        INSERT INTO\n            books (id,title,sort,timestamp,pubdate,series_index,author_sort,isbn,path)\n            SELECT id,title,sort,timestamp,timestamp,series_index,author_sort,isbn,path FROM books_backup;\n        DROP TABLE books_backup;\n\n        DROP VIEW meta;\n        CREATE VIEW meta AS\n        SELECT id, title,\n               (SELECT concat(name) FROM authors WHERE authors.id IN (SELECT author from books_authors_link WHERE book=books.id)) authors,\n               (SELECT name FROM publishers WHERE publishers.id IN (SELECT publisher from books_publishers_link WHERE book=books.id)) publisher,\n               (SELECT rating FROM ratings WHERE ratings.id IN (SELECT rating from books_ratings_link WHERE book=books.id)) rating,\n               timestamp,\n               (SELECT MAX(uncompressed_size) FROM data WHERE book=books.id) size,\n               (SELECT concat(name) FROM tags WHERE tags.id IN (SELECT tag from books_tags_link WHERE book=books.id)) tags,\n               (SELECT text FROM comments WHERE book=books.id) comments,\n               (SELECT name FROM series WHERE series.id IN (SELECT series FROM books_series_link WHERE book=books.id)) series,\n               series_index,\n               sort,\n               author_sort,\n               (SELECT concat(format) FROM data WHERE data.book=books.id) formats,\n               isbn,\n               path,\n               lccn,\n               pubdate,\n               flags\n        FROM books;\n        ')

    def upgrade_version_5(self):
        if False:
            for i in range(10):
                print('nop')
        'Update indexes/triggers for new books table'
        self.conn.executescript('\n        BEGIN TRANSACTION;\n        CREATE INDEX authors_idx ON books (author_sort COLLATE NOCASE);\n        CREATE INDEX books_idx ON books (sort COLLATE NOCASE);\n        CREATE TRIGGER books_delete_trg\n            AFTER DELETE ON books\n            BEGIN\n                DELETE FROM books_authors_link WHERE book=OLD.id;\n                DELETE FROM books_publishers_link WHERE book=OLD.id;\n                DELETE FROM books_ratings_link WHERE book=OLD.id;\n                DELETE FROM books_series_link WHERE book=OLD.id;\n                DELETE FROM books_tags_link WHERE book=OLD.id;\n                DELETE FROM data WHERE book=OLD.id;\n                DELETE FROM comments WHERE book=OLD.id;\n                DELETE FROM conversion_options WHERE book=OLD.id;\n        END;\n        CREATE TRIGGER books_insert_trg\n            AFTER INSERT ON books\n            BEGIN\n            UPDATE books SET sort=title_sort(NEW.title) WHERE id=NEW.id;\n        END;\n        CREATE TRIGGER books_update_trg\n            AFTER UPDATE ON books\n            BEGIN\n            UPDATE books SET sort=title_sort(NEW.title) WHERE id=NEW.id;\n        END;\n\n        UPDATE books SET sort=title_sort(title) WHERE sort IS NULL;\n\n        END TRANSACTION;\n        ')

    def upgrade_version_6(self):
        if False:
            print('Hello World!')
        'Show authors in order'
        self.conn.executescript('\n        BEGIN TRANSACTION;\n        DROP VIEW meta;\n        CREATE VIEW meta AS\n        SELECT id, title,\n               (SELECT sortconcat(bal.id, name) FROM books_authors_link AS bal JOIN authors ON(author = authors.id) WHERE book = books.id) authors,\n               (SELECT name FROM publishers WHERE publishers.id IN (SELECT publisher from books_publishers_link WHERE book=books.id)) publisher,\n               (SELECT rating FROM ratings WHERE ratings.id IN (SELECT rating from books_ratings_link WHERE book=books.id)) rating,\n               timestamp,\n               (SELECT MAX(uncompressed_size) FROM data WHERE book=books.id) size,\n               (SELECT concat(name) FROM tags WHERE tags.id IN (SELECT tag from books_tags_link WHERE book=books.id)) tags,\n               (SELECT text FROM comments WHERE book=books.id) comments,\n               (SELECT name FROM series WHERE series.id IN (SELECT series FROM books_series_link WHERE book=books.id)) series,\n               series_index,\n               sort,\n               author_sort,\n               (SELECT concat(format) FROM data WHERE data.book=books.id) formats,\n               isbn,\n               path,\n               lccn,\n               pubdate,\n               flags\n        FROM books;\n        END TRANSACTION;\n        ')

    def upgrade_version_7(self):
        if False:
            while True:
                i = 10
        'Add uuid column'
        self.conn.executescript('\n        BEGIN TRANSACTION;\n        ALTER TABLE books ADD COLUMN uuid TEXT;\n        DROP TRIGGER IF EXISTS books_insert_trg;\n        DROP TRIGGER IF EXISTS books_update_trg;\n        UPDATE books SET uuid=uuid4();\n\n        CREATE TRIGGER books_insert_trg AFTER INSERT ON books\n        BEGIN\n            UPDATE books SET sort=title_sort(NEW.title),uuid=uuid4() WHERE id=NEW.id;\n        END;\n\n        CREATE TRIGGER books_update_trg AFTER UPDATE ON books\n        BEGIN\n            UPDATE books SET sort=title_sort(NEW.title) WHERE id=NEW.id;\n        END;\n\n        DROP VIEW meta;\n        CREATE VIEW meta AS\n        SELECT id, title,\n               (SELECT sortconcat(bal.id, name) FROM books_authors_link AS bal JOIN authors ON(author = authors.id) WHERE book = books.id) authors,\n               (SELECT name FROM publishers WHERE publishers.id IN (SELECT publisher from books_publishers_link WHERE book=books.id)) publisher,\n               (SELECT rating FROM ratings WHERE ratings.id IN (SELECT rating from books_ratings_link WHERE book=books.id)) rating,\n               timestamp,\n               (SELECT MAX(uncompressed_size) FROM data WHERE book=books.id) size,\n               (SELECT concat(name) FROM tags WHERE tags.id IN (SELECT tag from books_tags_link WHERE book=books.id)) tags,\n               (SELECT text FROM comments WHERE book=books.id) comments,\n               (SELECT name FROM series WHERE series.id IN (SELECT series FROM books_series_link WHERE book=books.id)) series,\n               series_index,\n               sort,\n               author_sort,\n               (SELECT concat(format) FROM data WHERE data.book=books.id) formats,\n               isbn,\n               path,\n               lccn,\n               pubdate,\n               flags,\n               uuid\n        FROM books;\n\n        END TRANSACTION;\n        ')

    def upgrade_version_8(self):
        if False:
            print('Hello World!')
        'Add Tag Browser views'

        def create_tag_browser_view(table_name, column_name):
            if False:
                for i in range(10):
                    print('nop')
            self.conn.executescript('\n                DROP VIEW IF EXISTS tag_browser_{tn};\n                CREATE VIEW tag_browser_{tn} AS SELECT\n                    id,\n                    name,\n                    (SELECT COUNT(id) FROM books_{tn}_link WHERE {cn}={tn}.id) count\n                FROM {tn};\n                '.format(tn=table_name, cn=column_name))
        for tn in ('authors', 'tags', 'publishers', 'series'):
            cn = tn[:-1]
            if tn == 'series':
                cn = tn
            create_tag_browser_view(tn, cn)

    def upgrade_version_9(self):
        if False:
            for i in range(10):
                print('nop')
        'Add custom columns'
        self.conn.executescript('\n                CREATE TABLE custom_columns (\n                    id       INTEGER PRIMARY KEY AUTOINCREMENT,\n                    label    TEXT NOT NULL,\n                    name     TEXT NOT NULL,\n                    datatype TEXT NOT NULL,\n                    mark_for_delete   BOOL DEFAULT 0 NOT NULL,\n                    editable BOOL DEFAULT 1 NOT NULL,\n                    display  TEXT DEFAULT "{}" NOT NULL,\n                    is_multiple BOOL DEFAULT 0 NOT NULL,\n                    normalized BOOL NOT NULL,\n                    UNIQUE(label)\n                );\n                CREATE INDEX custom_columns_idx ON custom_columns (label);\n                CREATE INDEX IF NOT EXISTS formats_idx ON data (format);\n        ')

    def upgrade_version_10(self):
        if False:
            while True:
                i = 10
        'Add restricted Tag Browser views'

        def create_tag_browser_view(table_name, column_name, view_column_name):
            if False:
                for i in range(10):
                    print('nop')
            script = '\n                DROP VIEW IF EXISTS tag_browser_{tn};\n                CREATE VIEW tag_browser_{tn} AS SELECT\n                    id,\n                    {vcn},\n                    (SELECT COUNT(id) FROM books_{tn}_link WHERE {cn}={tn}.id) count\n                FROM {tn};\n                DROP VIEW IF EXISTS tag_browser_filtered_{tn};\n                CREATE VIEW tag_browser_filtered_{tn} AS SELECT\n                    id,\n                    {vcn},\n                    (SELECT COUNT(books_{tn}_link.id) FROM books_{tn}_link WHERE\n                        {cn}={tn}.id AND books_list_filter(book)) count\n                FROM {tn};\n                '.format(tn=table_name, cn=column_name, vcn=view_column_name)
            self.conn.executescript(script)
        for field in itervalues(self.field_metadata):
            if field['is_category'] and (not field['is_custom']) and ('link_column' in field):
                table = self.conn.get('SELECT name FROM sqlite_master WHERE type="table" AND name=?', ('books_%s_link' % field['table'],), all=False)
                if table is not None:
                    create_tag_browser_view(field['table'], field['link_column'], field['column'])

    def upgrade_version_11(self):
        if False:
            print('Hello World!')
        'Add average rating to tag browser views'

        def create_std_tag_browser_view(table_name, column_name, view_column_name, sort_column_name):
            if False:
                for i in range(10):
                    print('nop')
            script = '\n                DROP VIEW IF EXISTS tag_browser_{tn};\n                CREATE VIEW tag_browser_{tn} AS SELECT\n                    id,\n                    {vcn},\n                    (SELECT COUNT(id) FROM books_{tn}_link WHERE {cn}={tn}.id) count,\n                    (SELECT AVG(ratings.rating)\n                     FROM books_{tn}_link AS tl, books_ratings_link AS bl, ratings\n                     WHERE tl.{cn}={tn}.id AND bl.book=tl.book AND\n                     ratings.id = bl.rating AND ratings.rating <> 0) avg_rating,\n                     {scn} AS sort\n                FROM {tn};\n                DROP VIEW IF EXISTS tag_browser_filtered_{tn};\n                CREATE VIEW tag_browser_filtered_{tn} AS SELECT\n                    id,\n                    {vcn},\n                    (SELECT COUNT(books_{tn}_link.id) FROM books_{tn}_link WHERE\n                        {cn}={tn}.id AND books_list_filter(book)) count,\n                    (SELECT AVG(ratings.rating)\n                     FROM books_{tn}_link AS tl, books_ratings_link AS bl, ratings\n                     WHERE tl.{cn}={tn}.id AND bl.book=tl.book AND\n                     ratings.id = bl.rating AND ratings.rating <> 0 AND\n                     books_list_filter(bl.book)) avg_rating,\n                     {scn} AS sort\n                FROM {tn};\n\n                '.format(tn=table_name, cn=column_name, vcn=view_column_name, scn=sort_column_name)
            self.conn.executescript(script)

        def create_cust_tag_browser_view(table_name, link_table_name):
            if False:
                for i in range(10):
                    print('nop')
            script = '\n                DROP VIEW IF EXISTS tag_browser_{table};\n                CREATE VIEW tag_browser_{table} AS SELECT\n                    id,\n                    value,\n                    (SELECT COUNT(id) FROM {lt} WHERE value={table}.id) count,\n                    (SELECT AVG(r.rating)\n                     FROM {lt},\n                          books_ratings_link AS bl,\n                          ratings AS r\n                     WHERE {lt}.value={table}.id AND bl.book={lt}.book AND\n                           r.id = bl.rating AND r.rating <> 0) avg_rating,\n                     value AS sort\n                FROM {table};\n\n                DROP VIEW IF EXISTS tag_browser_filtered_{table};\n                CREATE VIEW tag_browser_filtered_{table} AS SELECT\n                    id,\n                    value,\n                    (SELECT COUNT({lt}.id) FROM {lt} WHERE value={table}.id AND\n                    books_list_filter(book)) count,\n                    (SELECT AVG(r.rating)\n                     FROM {lt},\n                          books_ratings_link AS bl,\n                          ratings AS r\n                     WHERE {lt}.value={table}.id AND bl.book={lt}.book AND\n                           r.id = bl.rating AND r.rating <> 0 AND\n                           books_list_filter(bl.book)) avg_rating,\n                     value AS sort\n                FROM {table};\n                '.format(lt=link_table_name, table=table_name)
            self.conn.executescript(script)
        for field in itervalues(self.field_metadata):
            if field['is_category'] and (not field['is_custom']) and ('link_column' in field):
                table = self.conn.get('SELECT name FROM sqlite_master WHERE type="table" AND name=?', ('books_%s_link' % field['table'],), all=False)
                if table is not None:
                    create_std_tag_browser_view(field['table'], field['link_column'], field['column'], field['category_sort'])
        db_tables = self.conn.get("SELECT name FROM sqlite_master\n                                     WHERE type='table'\n                                     ORDER BY name")
        tables = []
        for (table,) in db_tables:
            tables.append(table)
        for table in tables:
            link_table = 'books_%s_link' % table
            if table.startswith('custom_column_') and link_table in tables:
                create_cust_tag_browser_view(table, link_table)
        self.conn.execute('UPDATE authors SET sort=author_to_author_sort(name)')

    def upgrade_version_12(self):
        if False:
            print('Hello World!')
        'DB based preference store'
        script = '\n        DROP TABLE IF EXISTS preferences;\n        CREATE TABLE preferences(id INTEGER PRIMARY KEY,\n                                 key TEXT NON NULL,\n                                 val TEXT NON NULL,\n                                 UNIQUE(key));\n        '
        self.conn.executescript(script)

    def upgrade_version_13(self):
        if False:
            print('Hello World!')
        'Dirtied table for OPF metadata backups'
        script = '\n        DROP TABLE IF EXISTS metadata_dirtied;\n        CREATE TABLE metadata_dirtied(id INTEGER PRIMARY KEY,\n                             book INTEGER NOT NULL,\n                             UNIQUE(book));\n        INSERT INTO metadata_dirtied (book) SELECT id FROM books;\n        '
        self.conn.executescript(script)

    def upgrade_version_14(self):
        if False:
            print('Hello World!')
        'Cache has_cover'
        self.conn.execute('ALTER TABLE books ADD COLUMN has_cover BOOL DEFAULT 0')
        data = self.conn.get('SELECT id,path FROM books', all=True)

        def has_cover(path):
            if False:
                for i in range(10):
                    print('nop')
            if path:
                path = os.path.join(self.library_path, path.replace('/', os.sep), 'cover.jpg')
                return os.path.exists(path)
            return False
        ids = [(x[0],) for x in data if has_cover(x[1])]
        self.conn.executemany('UPDATE books SET has_cover=1 WHERE id=?', ids)

    def upgrade_version_15(self):
        if False:
            i = 10
            return i + 15
        'Remove commas from tags'
        self.conn.execute("UPDATE OR IGNORE tags SET name=REPLACE(name, ',', ';')")
        self.conn.execute("UPDATE OR IGNORE tags SET name=REPLACE(name, ',', ';;')")
        self.conn.execute("UPDATE OR IGNORE tags SET name=REPLACE(name, ',', '')")

    def upgrade_version_16(self):
        if False:
            return 10
        self.conn.executescript('\n        DROP TRIGGER IF EXISTS books_update_trg;\n        CREATE TRIGGER books_update_trg\n            AFTER UPDATE ON books\n            BEGIN\n            UPDATE books SET sort=title_sort(NEW.title)\n                         WHERE id=NEW.id AND OLD.title <> NEW.title;\n            END;\n        ')

    def upgrade_version_17(self):
        if False:
            return 10
        'custom book data table (for plugins)'
        script = '\n        DROP TABLE IF EXISTS books_plugin_data;\n        CREATE TABLE books_plugin_data(id INTEGER PRIMARY KEY,\n                                     book INTEGER NON NULL,\n                                     name TEXT NON NULL,\n                                     val TEXT NON NULL,\n                                     UNIQUE(book,name));\n        DROP TRIGGER IF EXISTS books_delete_trg;\n        CREATE TRIGGER books_delete_trg\n            AFTER DELETE ON books\n            BEGIN\n                DELETE FROM books_authors_link WHERE book=OLD.id;\n                DELETE FROM books_publishers_link WHERE book=OLD.id;\n                DELETE FROM books_ratings_link WHERE book=OLD.id;\n                DELETE FROM books_series_link WHERE book=OLD.id;\n                DELETE FROM books_tags_link WHERE book=OLD.id;\n                DELETE FROM data WHERE book=OLD.id;\n                DELETE FROM comments WHERE book=OLD.id;\n                DELETE FROM conversion_options WHERE book=OLD.id;\n                DELETE FROM books_plugin_data WHERE book=OLD.id;\n        END;\n        '
        self.conn.executescript(script)

    def upgrade_version_18(self):
        if False:
            return 10
        '\n        Add a library UUID.\n        Add an identifiers table.\n        Add a languages table.\n        Add a last_modified column.\n        NOTE: You cannot downgrade after this update, if you do\n        any changes you make to book isbns will be lost.\n        '
        script = '\n        DROP TABLE IF EXISTS library_id;\n        CREATE TABLE library_id ( id   INTEGER PRIMARY KEY,\n                                  uuid TEXT NOT NULL,\n                                  UNIQUE(uuid)\n        );\n\n        DROP TABLE IF EXISTS identifiers;\n        CREATE TABLE identifiers  ( id     INTEGER PRIMARY KEY,\n                                    book   INTEGER NON NULL,\n                                    type   TEXT NON NULL DEFAULT "isbn" COLLATE NOCASE,\n                                    val    TEXT NON NULL COLLATE NOCASE,\n                                    UNIQUE(book, type)\n        );\n\n        DROP TABLE IF EXISTS languages;\n        CREATE TABLE languages    ( id        INTEGER PRIMARY KEY,\n                                    lang_code TEXT NON NULL COLLATE NOCASE,\n                                    UNIQUE(lang_code)\n        );\n\n        DROP TABLE IF EXISTS books_languages_link;\n        CREATE TABLE books_languages_link ( id INTEGER PRIMARY KEY,\n                                            book INTEGER NOT NULL,\n                                            lang_code INTEGER NOT NULL,\n                                            item_order INTEGER NOT NULL DEFAULT 0,\n                                            UNIQUE(book, lang_code)\n        );\n\n        DROP TRIGGER IF EXISTS fkc_delete_on_languages;\n        CREATE TRIGGER fkc_delete_on_languages\n        BEFORE DELETE ON languages\n        BEGIN\n            SELECT CASE\n                WHEN (SELECT COUNT(id) FROM books_languages_link WHERE lang_code=OLD.id) > 0\n                THEN RAISE(ABORT, \'Foreign key violation: language is still referenced\')\n            END;\n        END;\n\n        DROP TRIGGER IF EXISTS fkc_delete_on_languages_link;\n        CREATE TRIGGER fkc_delete_on_languages_link\n        BEFORE INSERT ON books_languages_link\n        BEGIN\n          SELECT CASE\n              WHEN (SELECT id from books WHERE id=NEW.book) IS NULL\n              THEN RAISE(ABORT, \'Foreign key violation: book not in books\')\n              WHEN (SELECT id from languages WHERE id=NEW.lang_code) IS NULL\n              THEN RAISE(ABORT, \'Foreign key violation: lang_code not in languages\')\n          END;\n        END;\n\n        DROP TRIGGER IF EXISTS fkc_update_books_languages_link_a;\n        CREATE TRIGGER fkc_update_books_languages_link_a\n        BEFORE UPDATE OF book ON books_languages_link\n        BEGIN\n            SELECT CASE\n                WHEN (SELECT id from books WHERE id=NEW.book) IS NULL\n                THEN RAISE(ABORT, \'Foreign key violation: book not in books\')\n            END;\n        END;\n        DROP TRIGGER IF EXISTS fkc_update_books_languages_link_b;\n        CREATE TRIGGER fkc_update_books_languages_link_b\n        BEFORE UPDATE OF lang_code ON books_languages_link\n        BEGIN\n            SELECT CASE\n                WHEN (SELECT id from languages WHERE id=NEW.lang_code) IS NULL\n                THEN RAISE(ABORT, \'Foreign key violation: lang_code not in languages\')\n            END;\n        END;\n\n        DROP INDEX IF EXISTS books_languages_link_aidx;\n        CREATE INDEX books_languages_link_aidx ON books_languages_link (lang_code);\n        DROP INDEX IF EXISTS books_languages_link_bidx;\n        CREATE INDEX books_languages_link_bidx ON books_languages_link (book);\n        DROP INDEX IF EXISTS languages_idx;\n        CREATE INDEX languages_idx ON languages (lang_code COLLATE NOCASE);\n\n        DROP TRIGGER IF EXISTS books_delete_trg;\n        CREATE TRIGGER books_delete_trg\n            AFTER DELETE ON books\n            BEGIN\n                DELETE FROM books_authors_link WHERE book=OLD.id;\n                DELETE FROM books_publishers_link WHERE book=OLD.id;\n                DELETE FROM books_ratings_link WHERE book=OLD.id;\n                DELETE FROM books_series_link WHERE book=OLD.id;\n                DELETE FROM books_tags_link WHERE book=OLD.id;\n                DELETE FROM books_languages_link WHERE book=OLD.id;\n                DELETE FROM data WHERE book=OLD.id;\n                DELETE FROM comments WHERE book=OLD.id;\n                DELETE FROM conversion_options WHERE book=OLD.id;\n                DELETE FROM books_plugin_data WHERE book=OLD.id;\n                DELETE FROM identifiers WHERE book=OLD.id;\n        END;\n\n        INSERT INTO identifiers (book, val) SELECT id,isbn FROM books WHERE isbn;\n\n        ALTER TABLE books ADD COLUMN last_modified TIMESTAMP NOT NULL DEFAULT "%s";\n\n        ' % isoformat(DEFAULT_DATE, sep=' ')
        self.conn.executescript(script)

    def upgrade_version_19(self):
        if False:
            i = 10
            return i + 15
        recipes = self.conn.get('SELECT id,title,script FROM feeds')
        if recipes:
            from calibre.web.feeds.recipes import custom_recipes, custom_recipe_filename
            bdir = os.path.dirname(custom_recipes.file_path)
            for (id_, title, script) in recipes:
                existing = frozenset(map(int, custom_recipes))
                if id_ in existing:
                    id_ = max(existing) + 1000
                id_ = str(id_)
                fname = custom_recipe_filename(id_, title)
                custom_recipes[id_] = (title, fname)
                if isinstance(script, str):
                    script = script.encode('utf-8')
                with open(os.path.join(bdir, fname), 'wb') as f:
                    f.write(script)

    def upgrade_version_20(self):
        if False:
            return 10
        '\n        Add a link column to the authors table.\n        '
        script = '\n        BEGIN TRANSACTION;\n        ALTER TABLE authors ADD COLUMN link TEXT NOT NULL DEFAULT "";\n        '
        self.conn.executescript(script)