from dagster import MetadataValue, graph, job, op

@op
def my_op():
    if False:
        while True:
            i = 10
    return 'Hello World!'

@job(metadata={'owner': 'data team', 'docs': MetadataValue.url('https://docs.dagster.io')})
def my_job_with_metadata():
    if False:
        for i in range(10):
            print('nop')
    my_op()

@graph
def my_graph():
    if False:
        while True:
            i = 10
    my_op()
my_second_job_with_metadata = my_graph.to_job(metadata={'owner': 'api team', 'docs': MetadataValue.url('https://docs.dagster.io')})

@job(tags={'foo': 'bar'})
def my_job_with_tags():
    if False:
        i = 10
        return i + 15
    my_op()
my_second_job_with_tags = my_graph.to_job(tags={'foo': 'bar'})