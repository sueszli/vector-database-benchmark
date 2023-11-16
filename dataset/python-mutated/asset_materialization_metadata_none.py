from dagster import Output, asset

@asset
def table1() -> Output[None]:
    if False:
        print('Hello World!')
    ...
    return Output(None, metadata={'num_rows': 25})