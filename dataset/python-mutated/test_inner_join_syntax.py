import unittest
from pony.orm import *
from pony import orm
from pony.orm.tests import setup_database, teardown_database, only_for
db = Database()

class Genre(db.Entity):
    name = orm.Optional(str)
    artists = orm.Set('Artist')
    favorite = orm.Optional(bool)
    index = orm.Optional(int)

class Hobby(db.Entity):
    name = orm.Required(str)
    artists = orm.Set('Artist')

class Artist(db.Entity):
    name = orm.Required(str)
    age = orm.Optional(int)
    hobbies = orm.Set(Hobby)
    genres = orm.Set(Genre)
pony.options.INNER_JOIN_SYNTAX = True

@only_for('sqlite')
class TestJoin(unittest.TestCase):
    exclude_fixtures = {'test': ['clear_tables']}

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        setup_database(db)
        with orm.db_session:
            pop = Genre(name='pop')
            rock = Genre(name='rock')
            Artist(name='Sia', age=40, genres=[pop, rock])
            Artist(name='Lady GaGa', age=30, genres=[pop])

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        teardown_database(db)

    @db_session
    def test_join_1(self):
        if False:
            print('Hello World!')
        result = select((g.id for g in db.Genre for a in g.artists if a.name.startswith('S')))[:]
        self.assertEqual(db.last_sql, 'SELECT DISTINCT "g"."id"\nFROM "Genre" "g"\n  INNER JOIN "Artist_Genre" "t-1"\n    ON "g"."id" = "t-1"."genre"\n  INNER JOIN "Artist" "a"\n    ON "t-1"."artist" = "a"."id"\nWHERE "a"."name" LIKE \'S%\'')

    @db_session
    def test_join_2(self):
        if False:
            for i in range(10):
                print('nop')
        result = select((g.id for g in db.Genre for a in db.Artist if JOIN(a in g.artists) and a.name.startswith('S')))[:]
        self.assertEqual(db.last_sql, 'SELECT DISTINCT "g"."id"\nFROM "Genre" "g"\n  INNER JOIN "Artist_Genre" "t-1"\n    ON "g"."id" = "t-1"."genre", "Artist" "a"\nWHERE "t-1"."artist" = "a"."id"\n  AND "a"."name" LIKE \'S%\'')

    @db_session
    def test_join_3(self):
        if False:
            for i in range(10):
                print('nop')
        result = select((g.id for g in db.Genre for x in db.Artist for a in db.Artist if JOIN(a in g.artists) and a.name.startswith('S') and (g.id == x.id)))[:]
        self.assertEqual(db.last_sql, 'SELECT DISTINCT "g"."id"\nFROM "Genre" "g"\n  INNER JOIN "Artist_Genre" "t-1"\n    ON "g"."id" = "t-1"."genre", "Artist" "x", "Artist" "a"\nWHERE "t-1"."artist" = "a"."id"\n  AND "a"."name" LIKE \'S%\'\n  AND "g"."id" = "x"."id"')

    @db_session
    def test_join_4(self):
        if False:
            return 10
        result = select((g.id for g in db.Genre for a in db.Artist for x in db.Artist if JOIN(a in g.artists) and a.name.startswith('S') and (g.id == x.id)))[:]
        self.assertEqual(db.last_sql, 'SELECT DISTINCT "g"."id"\nFROM "Genre" "g"\n  INNER JOIN "Artist_Genre" "t-1"\n    ON "g"."id" = "t-1"."genre", "Artist" "a", "Artist" "x"\nWHERE "t-1"."artist" = "a"."id"\n  AND "a"."name" LIKE \'S%\'\n  AND "g"."id" = "x"."id"')
if __name__ == '__main__':
    unittest.main()