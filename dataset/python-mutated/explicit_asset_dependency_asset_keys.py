from dagster import AssetIn, asset

@asset(ins={'upstream': AssetIn(key='upstream_asset')})
def downstream_asset(upstream):
    if False:
        print('Hello World!')
    return upstream + [4]

@asset(ins={'upstream': AssetIn(key=['some_db_schema', 'upstream_asset'])})
def another_downstream_asset(upstream):
    if False:
        while True:
            i = 10
    return upstream + [10]