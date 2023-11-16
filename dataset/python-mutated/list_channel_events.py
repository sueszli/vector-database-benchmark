"""Google Cloud Live Stream sample for listing all events for a channel.
Example usage:
    python list_channel_events.py --project_id <project-id> --location <location> --channel_id <channel-id>
"""
import argparse
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient, pagers

def list_channel_events(project_id: str, location: str, channel_id: str) -> pagers.ListEventsPager:
    if False:
        while True:
            i = 10
    'Lists all events for a channel.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the channel.\n        channel_id: The user-defined channel ID.'
    client = LivestreamServiceClient()
    parent = f'projects/{project_id}/locations/{location}/channels/{channel_id}'
    page_result = client.list_events(parent=parent)
    print('Events:')
    responses = []
    for response in page_result:
        print(response.name)
        responses.append(response)
    return responses
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the channel.', required=True)
    parser.add_argument('--channel_id', help='The user-defined channel ID.', required=True)
    args = parser.parse_args()
    list_channel_events(args.project_id, args.location, args.channel_id)