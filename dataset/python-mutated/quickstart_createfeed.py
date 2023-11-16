import argparse

def create_feed(project_id, feed_id, asset_names, topic, content_type):
    if False:
        print('Hello World!')
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    parent = f'projects/{project_id}'
    feed = asset_v1.Feed()
    feed.asset_names.extend(asset_names)
    feed.feed_output_config.pubsub_destination.topic = topic
    feed.content_type = content_type
    response = client.create_feed(request={'parent': parent, 'feed_id': feed_id, 'feed': feed})
    print(f'feed: {response}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID')
    parser.add_argument('feed_id', help='Feed ID you want to create')
    parser.add_argument('asset_names', help='List of asset names the feed listen to')
    parser.add_argument('topic', help='Topic name of the feed')
    parser.add_argument('content_type', help='Content type of the feed')
    args = parser.parse_args()
    create_feed(args.project_id, args.feed_id, args.asset_names, args.topic, args.content_type)