import pickle
from dagster import asset

@asset
def upstream_asset():
    if False:
        for i in range(10):
            print('nop')
    with open('upstream_asset.pkl', 'wb') as f:
        pickle.dump([1, 2, 3], f)

@asset(deps=[upstream_asset])
def downstream_asset():
    if False:
        for i in range(10):
            print('nop')
    with open('upstream_asset.pkl', 'wb') as f:
        data = pickle.load(f)
    with open('downstream_asset.pkl', 'wb') as f:
        pickle.dump(f, data + [4])