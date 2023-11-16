from dagster import graph

@graph
def graph_one():
    if False:
        for i in range(10):
            print('nop')
    pass

@graph
def graph_two():
    if False:
        print('Hello World!')
    pass