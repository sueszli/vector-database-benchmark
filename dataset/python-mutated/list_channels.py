"""Google Cloud Live Stream sample for listing all channels in a location.
Example usage:
    python list_channels.py --project_id <project-id> --location <location>
"""
import argparse
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient, pagers

def list_channels(project_id: str, location: str) -> pagers.ListChannelsPager:
    if False:
        for i in range(10):
            print('nop')
    'Lists all channels in a location.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the channels.'
    client = LivestreamServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    page_result = client.list_channels(parent=parent)
    print('Channels:')
    responses = []
    for response in page_result:
        print(response.name)
        responses.append(response)
    return responses
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the channels.', required=True)
    args = parser.parse_args()
    list_channels(args.project_id, args.location)