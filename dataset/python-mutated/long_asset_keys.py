from dagster import AssetIn, asset
key_prefix1 = ['s3', 'superdomain_1', 'subdomain_1', 'subsubdomain_1']

@asset(key_prefix=key_prefix1)
def asset1():
    if False:
        print('Hello World!')
    pass

@asset(key_prefix=['s3', 'superdomain_2', 'subdomain_2', 'subsubdomain_2'], ins={'asset1': AssetIn(key_prefix=key_prefix1)})
def asset2(asset1):
    if False:
        while True:
            i = 10
    assert asset1 is None