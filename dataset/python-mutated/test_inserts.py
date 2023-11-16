import timeit

def main():
    if False:
        while True:
            i = 10
    setup = "\nfrom pymongo import MongoClient\n\nconnection = MongoClient(w=1)\nconnection.drop_database('mongoengine_benchmark_test')\n"
    stmt = '\ndb = connection.mongoengine_benchmark_test\nnoddy = db.noddy\n\nfor i in range(10000):\n    example = {\'fields\': {}}\n    for j in range(20):\n        example[\'fields\']["key"+str(j)] = "value "+str(j)\n\n    noddy.insert_one(example)\n\nmyNoddys = noddy.find()\n[n for n in myNoddys]  # iterate\n'
    print('-' * 100)
    print('PyMongo: Creating 10000 dictionaries (write_concern={"w": 1}).')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{t.timeit(1)}s')
    stmt = '\nfrom pymongo import WriteConcern\n\ndb = connection.mongoengine_benchmark_test\nnoddy = db.noddy.with_options(write_concern=WriteConcern(w=0))\n\nfor i in range(10000):\n    example = {\'fields\': {}}\n    for j in range(20):\n        example[\'fields\']["key"+str(j)] = "value "+str(j)\n\n    noddy.insert_one(example)\n\nmyNoddys = noddy.find()\n[n for n in myNoddys]  # iterate\n'
    print('-' * 100)
    print('PyMongo: Creating 10000 dictionaries (write_concern={"w": 0}).')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{t.timeit(1)}s')
    setup = '\nfrom pymongo import MongoClient\n\nconnection = MongoClient()\nconnection.drop_database(\'mongoengine_benchmark_test\')\nconnection.close()\n\nfrom mongoengine import Document, DictField, connect\nconnect("mongoengine_benchmark_test", w=1)\n\nclass Noddy(Document):\n    fields = DictField()\n'
    stmt = '\nfor i in range(10000):\n    noddy = Noddy()\n    for j in range(20):\n        noddy.fields["key"+str(j)] = "value "+str(j)\n    noddy.save()\n\nmyNoddys = Noddy.objects()\n[n for n in myNoddys]  # iterate\n'
    print('-' * 100)
    print('MongoEngine: Creating 10000 dictionaries (write_concern={"w": 1}).')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{t.timeit(1)}s')
    stmt = '\nfor i in range(10000):\n    noddy = Noddy()\n    fields = {}\n    for j in range(20):\n        fields["key"+str(j)] = "value "+str(j)\n    noddy.fields = fields\n    noddy.save()\n\nmyNoddys = Noddy.objects()\n[n for n in myNoddys]  # iterate\n'
    print('-' * 100)
    print('MongoEngine: Creating 10000 dictionaries (using a single field assignment).')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{t.timeit(1)}s')
    stmt = '\nfor i in range(10000):\n    noddy = Noddy()\n    for j in range(20):\n        noddy.fields["key"+str(j)] = "value "+str(j)\n    noddy.save(write_concern={"w": 0})\n\nmyNoddys = Noddy.objects()\n[n for n in myNoddys] # iterate\n'
    print('-' * 100)
    print('MongoEngine: Creating 10000 dictionaries (write_concern={"w": 0}).')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{t.timeit(1)}s')
    stmt = '\nfor i in range(10000):\n    noddy = Noddy()\n    for j in range(20):\n        noddy.fields["key"+str(j)] = "value "+str(j)\n    noddy.save(write_concern={"w": 0}, validate=False)\n\nmyNoddys = Noddy.objects()\n[n for n in myNoddys] # iterate\n'
    print('-' * 100)
    print('MongoEngine: Creating 10000 dictionaries (write_concern={"w": 0}, validate=False).')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{t.timeit(1)}s')
    stmt = '\nfor i in range(10000):\n    noddy = Noddy()\n    for j in range(20):\n        noddy.fields["key"+str(j)] = "value "+str(j)\n    noddy.save(force_insert=True, write_concern={"w": 0}, validate=False)\n\nmyNoddys = Noddy.objects()\n[n for n in myNoddys] # iterate\n'
    print('-' * 100)
    print('MongoEngine: Creating 10000 dictionaries (force_insert=True, write_concern={"w": 0}, validate=False).')
    t = timeit.Timer(stmt=stmt, setup=setup)
    print(f'{t.timeit(1)}s')
if __name__ == '__main__':
    main()