from dagster import AssetExecutionContext, InitResourceContext, OpExecutionContext, asset, job, op, repository, resource

class MyDatabaseConnection:

    def __init__(self, url):
        if False:
            i = 10
            return i + 15
        self.url = url

@op(config_schema={'person_name': str})
def op_using_config(context: OpExecutionContext):
    if False:
        print('Hello World!')
    return f"hello {context.op_config['person_name']}"

@asset(config_schema={'person_name': str})
def asset_using_config(context: AssetExecutionContext):
    if False:
        while True:
            i = 10
    return f"hello {context.op_config['person_name']}"

@resource(config_schema={'url': str})
def resource_using_config(context: InitResourceContext):
    if False:
        while True:
            i = 10
    return MyDatabaseConnection(context.resource_config['url'])

@job
def job_using_config():
    if False:
        while True:
            i = 10
    op_using_config()

@repository
def repo():
    if False:
        while True:
            i = 10
    return [job_using_config]