from dagster import graph

@graph
def graph_one():
    if False:
        while True:
            i = 10
    pass