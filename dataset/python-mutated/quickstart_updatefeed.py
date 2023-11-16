import argparse

def update_feed(feed_name, topic):
    if False:
        print('Hello World!')
    from google.cloud import asset_v1
    from google.protobuf import field_mask_pb2
    client = asset_v1.AssetServiceClient()
    feed = asset_v1.Feed()
    feed.name = feed_name
    feed.feed_output_config.pubsub_destination.topic = topic
    update_mask = field_mask_pb2.FieldMask()
    update_mask.paths.append('feed_output_config.pubsub_destination.topic')
    response = client.update_feed(request={'feed': feed, 'update_mask': update_mask})
    print(f'updated_feed: {response}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('feed_name', help='Feed Name you want to update')
    parser.add_argument('topic', help='Topic name you want to update with')
    args = parser.parse_args()
    update_feed(args.feed_name, args.topic)