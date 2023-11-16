import argparse

def get_feed(feed_name):
    if False:
        for i in range(10):
            print('nop')
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    response = client.get_feed(request={'name': feed_name})
    print(f'gotten_feed: {response}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('feed_name', help='Feed Name you want to get')
    args = parser.parse_args()
    get_feed(args.feed_name)