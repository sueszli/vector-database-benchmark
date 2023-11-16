class MockDatabase:

    def execute(self, query: str):
        if False:
            for i in range(10):
                print('nop')
        pass

def get_database_connection():
    if False:
        return 10
    return MockDatabase()
from dagster import In, Nothing, job, op

@op
def create_table_1():
    if False:
        print('Hello World!')
    get_database_connection().execute('create table_1 as select * from some_source_table')

@op(ins={'start': In(Nothing)})
def create_table_2():
    if False:
        i = 10
        return i + 15
    get_database_connection().execute('create table_2 as select * from table_1')

@job
def nothing_dependency():
    if False:
        i = 10
        return i + 15
    create_table_2(start=create_table_1())