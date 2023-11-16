def execute_query(query):
    if False:
        print('Hello World!')
    del query
from dagster import asset

@asset
def sugary_cereals() -> None:
    if False:
        i = 10
        return i + 15
    execute_query('CREATE TABLE sugary_cereals AS SELECT * FROM cereals')

@asset(deps=[sugary_cereals])
def shopping_list() -> None:
    if False:
        print('Hello World!')
    execute_query('CREATE TABLE shopping_list AS SELECT * FROM sugary_cereals')