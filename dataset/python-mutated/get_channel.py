"""Google Cloud Live Stream sample for getting a channel.
Example usage:
    python get_channel.py --project_id <project-id> --location <location> --channel_id <channel-id>
"""
import argparse
from google.cloud.video import live_stream_v1
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient

def get_channel(project_id: str, location: str, channel_id: str) -> live_stream_v1.types.Channel:
    if False:
        while True:
            i = 10
    'Gets a channel.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the channel.\n        channel_id: The user-defined channel ID.'
    client = LivestreamServiceClient()
    name = f'projects/{project_id}/locations/{location}/channels/{channel_id}'
    response = client.get_channel(name=name)
    print(f'Channel: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the channel.', required=True)
    parser.add_argument('--channel_id', help='The user-defined channel ID.', required=True)
    args = parser.parse_args()
    get_channel(args.project_id, args.location, args.channel_id)