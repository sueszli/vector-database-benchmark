"""Google Cloud Live Stream sample for deleting a channel.
Example usage:
    python delete_channel.py --project_id <project-id> --location <location> --channel_id <channel-id>
"""
import argparse
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient
from google.protobuf import empty_pb2 as empty

def delete_channel(project_id: str, location: str, channel_id: str) -> empty.Empty:
    if False:
        i = 10
        return i + 15
    'Deletes a channel.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the channel.\n        channel_id: The user-defined channel ID.'
    client = LivestreamServiceClient()
    name = f'projects/{project_id}/locations/{location}/channels/{channel_id}'
    operation = client.delete_channel(name=name)
    response = operation.result(600)
    print('Deleted channel')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the channel.', required=True)
    parser.add_argument('--channel_id', help='The user-defined channel ID.', required=True)
    args = parser.parse_args()
    delete_channel(args.project_id, args.location, args.channel_id)