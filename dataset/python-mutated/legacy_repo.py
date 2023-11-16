from dagster import graph, op, pipeline, repository, solid
from dagster_graphql import DagsterGraphQLClient

@solid
def my_solid():
    if False:
        print('Hello World!')
    return 5

@solid
def ingest_solid(x):
    if False:
        while True:
            i = 10
    return x + 5

@pipeline
def the_pipeline():
    if False:
        i = 10
        return i + 15
    ingest_solid(my_solid())

@op
def my_op():
    if False:
        while True:
            i = 10
    return 5

@op
def ingest(x):
    if False:
        for i in range(10):
            print('nop')
    return x + 5

@graph
def basic():
    if False:
        print('Hello World!')
    ingest(my_op())

@solid
def ping_dagit():
    if False:
        while True:
            i = 10
    client = DagsterGraphQLClient('dagster_webserver', port_number=3000)
    return client._execute('{__typename}')

@pipeline
def test_graphql():
    if False:
        while True:
            i = 10
    ping_dagit()
the_job = basic.to_job(name='the_job')

@repository
def basic_repo():
    if False:
        print('Hello World!')
    return [the_job, the_pipeline, test_graphql]