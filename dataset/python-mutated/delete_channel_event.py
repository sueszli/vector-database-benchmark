"""Google Cloud Live Stream sample for deleting a channel event.
Example usage:
    python delete_channel_event.py --project_id <project-id> --location <location>         --channel_id <channel-id> --event_id <event-id>
"""
import argparse
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient

def delete_channel_event(project_id: str, location: str, channel_id: str, event_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Deletes a channel event.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the channel.\n        channel_id: The user-defined channel ID.\n        event_id: The user-defined event ID.'
    client = LivestreamServiceClient()
    name = f'projects/{project_id}/locations/{location}/channels/{channel_id}/events/{event_id}'
    response = client.delete_event(name=name)
    print('Deleted channel event')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the channel.', required=True)
    parser.add_argument('--channel_id', help='The user-defined channel ID.', required=True)
    parser.add_argument('--event_id', help='The user-defined event ID.', required=True)
    args = parser.parse_args()
    delete_channel_event(args.project_id, args.location, args.channel_id, args.event_id)