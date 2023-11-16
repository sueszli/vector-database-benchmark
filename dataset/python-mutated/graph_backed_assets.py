from dagster import graph_asset, op

@op
def hello():
    if False:
        while True:
            i = 10
    return 'hello'

@op
def world(hello):
    if False:
        return 10
    return hello + 'world'

@graph_asset
def graph_backed_asset():
    if False:
        print('Hello World!')
    return world(hello())