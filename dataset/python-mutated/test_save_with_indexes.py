import timeit

def main():
    if False:
        for i in range(10):
            print('nop')
    setup = '\nfrom pymongo import MongoClient\n\nconnection = MongoClient()\nconnection.drop_database("mongoengine_benchmark_test")\nconnection.close()\n\nfrom mongoengine import connect, Document, IntField, StringField\nconnect("mongoengine_benchmark_test", w=1)\n\nclass User0(Document):\n    name = StringField()\n    age = IntField()\n\nclass User1(Document):\n    name = StringField()\n    age = IntField()\n    meta = {"indexes": [["name"]]}\n\nclass User2(Document):\n    name = StringField()\n    age = IntField()\n    meta = {"indexes": [["name", "age"]]}\n\nclass User3(Document):\n    name = StringField()\n    age = IntField()\n    meta = {"indexes": [["name"]], "auto_create_index_on_save": True}\n\nclass User4(Document):\n    name = StringField()\n    age = IntField()\n    meta = {"indexes": [["name", "age"]], "auto_create_index_on_save": True}\n'
    stmt = '\nfor i in range(10000):\n    User0(name="Nunu", age=9).save()\n'
    print('-' * 80)
    print('Save 10000 documents with 0 indexes.')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{min(t.repeat(repeat=3, number=1))}s')
    stmt = '\nfor i in range(10000):\n    User1(name="Nunu", age=9).save()\n'
    print('-' * 80)
    print('Save 10000 documents with 1 index.')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{min(t.repeat(repeat=3, number=1))}s')
    stmt = '\nfor i in range(10000):\n    User2(name="Nunu", age=9).save()\n'
    print('-' * 80)
    print('Save 10000 documents with 2 indexes.')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{min(t.repeat(repeat=3, number=1))}s')
    stmt = '\nfor i in range(10000):\n    User3(name="Nunu", age=9).save()\n'
    print('-' * 80)
    print('Save 10000 documents with 1 index (auto_create_index_on_save=True).')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{min(t.repeat(repeat=3, number=1))}s')
    stmt = '\nfor i in range(10000):\n    User4(name="Nunu", age=9).save()\n'
    print('-' * 80)
    print('Save 10000 documents with 2 indexes (auto_create_index_on_save=True).')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{min(t.repeat(repeat=3, number=1))}s')
if __name__ == '__main__':
    main()