class MockDatabase:

    def execute(self, query: str):
        if False:
            while True:
                i = 10
        pass

def get_database_connection():
    if False:
        for i in range(10):
            print('nop')
    return MockDatabase()
from dagster import In, Nothing, graph, op

@op
def create_table_1():
    if False:
        return 10
    get_database_connection().execute('create table_1 as select * from some_source_table')

@op(ins={'start': In(Nothing)})
def create_table_2():
    if False:
        i = 10
        return i + 15
    get_database_connection().execute('create table_2 as select * from table_1')

@graph
def nothing_dependency():
    if False:
        while True:
            i = 10
    create_table_2(start=create_table_1())