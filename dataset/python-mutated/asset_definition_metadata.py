from dagster import asset

@asset(metadata={'owner': 'alice@mycompany.com', 'priority': 'high'})
def my_asset():
    if False:
        print('Hello World!')
    return 5