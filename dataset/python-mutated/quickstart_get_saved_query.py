import argparse

def get_saved_query(saved_query_name):
    if False:
        while True:
            i = 10
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    response = client.get_saved_query(request={'name': saved_query_name})
    print(f'gotten_saved_query: {response}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('saved_query_name', help='SavedQuery Name you want to get')
    args = parser.parse_args()
    get_saved_query(args.saved_query_name)