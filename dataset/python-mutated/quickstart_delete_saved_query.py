import argparse

def delete_saved_query(saved_query_name):
    if False:
        while True:
            i = 10
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    client.delete_saved_query(request={'name': saved_query_name})
    print('deleted_saved_query')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('saved_query_name', help='SavedQuery name you want to delete')
    args = parser.parse_args()
    delete_saved_query(args.saved_query_name)