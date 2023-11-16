from dagster import graph, op

@op
def do_something():
    if False:
        i = 10
        return i + 15
    pass

@graph
def do_stuff():
    if False:
        i = 10
        return i + 15
    do_something()