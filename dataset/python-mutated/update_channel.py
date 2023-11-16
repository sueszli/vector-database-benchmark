"""Google Cloud Live Stream sample for updating a channel with a different input.
Example usage:
    python update_channel.py --project_id <project-id> --location <location>         --channel_id <channel-id> --input_id <input-id>
"""
import argparse
from google.cloud.video import live_stream_v1
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient
from google.protobuf import field_mask_pb2 as field_mask

def update_channel(project_id: str, location: str, channel_id: str, input_id: str) -> live_stream_v1.types.Channel:
    if False:
        i = 10
        return i + 15
    'Updates a channel.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the channel.\n        channel_id: The user-defined channel ID.\n        input_id: The user-defined input ID for the new input.'
    client = LivestreamServiceClient()
    input = f'projects/{project_id}/locations/{location}/inputs/{input_id}'
    name = f'projects/{project_id}/locations/{location}/channels/{channel_id}'
    channel = live_stream_v1.types.Channel(name=name, input_attachments=[live_stream_v1.types.InputAttachment(key='updated-input', input=input)])
    update_mask = field_mask.FieldMask(paths=['input_attachments'])
    operation = client.update_channel(channel=channel, update_mask=update_mask)
    response = operation.result(600)
    print(f'Updated channel: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location in of the channel.', required=True)
    parser.add_argument('--channel_id', help='The user-defined channel ID.', required=True)
    parser.add_argument('--input_id', help='The user-defined input ID.', required=True)
    args = parser.parse_args()
    update_channel(args.project_id, args.location, args.channel_id, args.input_id)