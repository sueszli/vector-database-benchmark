import argparse

def delete_feed(feed_name):
    if False:
        i = 10
        return i + 15
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    client.delete_feed(request={'name': feed_name})
    print('deleted_feed')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('feed_name', help='Feed name you want to delete')
    args = parser.parse_args()
    delete_feed(args.feed_name)