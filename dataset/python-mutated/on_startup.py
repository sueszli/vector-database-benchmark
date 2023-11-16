"""Provides an example of attaching an action on hug server startup"""
import hug
data = []

@hug.startup()
def add_data(api):
    if False:
        while True:
            i = 10
    'Adds initial data to the api on startup'
    data.append("It's working")

@hug.startup()
def add_more_data(api):
    if False:
        i = 10
        return i + 15
    'Adds initial data to the api on startup'
    data.append('Even subsequent calls')

@hug.cli()
@hug.get()
def test():
    if False:
        while True:
            i = 10
    'Returns all stored data'
    return data