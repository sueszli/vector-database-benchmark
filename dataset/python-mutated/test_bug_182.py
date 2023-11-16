import unittest
from pony.orm import *
from pony import orm
from pony.orm.tests import setup_database, teardown_database
db = Database()

class User(db.Entity):
    name = Required(str)
    servers = Set('Server')

class Worker(User):
    pass

class Admin(Worker):
    pass

class Server(db.Entity):
    name = Required(str)
    user = Optional(User)

class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        setup_database(db)
        with orm.db_session:
            Server(name='s1.example.com', user=User(name='Alex'))
            Server(name='s2.example.com', user=Worker(name='John'))
            Server(name='free.example.com', user=None)

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        teardown_database(db)

    @db_session
    def test(self):
        if False:
            return 10
        qu = left_join(((s.name, s.user.name) for s in db.Server))[:]
        for (server, user) in qu:
            if user is None:
                break
        else:
            self.fail()