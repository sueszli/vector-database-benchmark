"""Google Cloud Live Stream sample for creating a channel. A channel resource
    represents the processor that performs a user-defined "streaming" operation.
Example usage:
    python create_channel.py --project_id <project-id> --location <location>         --channel_id <channel-id> --input_id <input-id> --output_uri <uri>
"""
import argparse
from google.cloud.video import live_stream_v1
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient
from google.protobuf import duration_pb2 as duration

def create_channel(project_id: str, location: str, channel_id: str, input_id: str, output_uri: str) -> live_stream_v1.types.Channel:
    if False:
        return 10
    'Creates a channel.\n    Args:\n        project_id: The GCP project ID.\n        location: The location in which to create the channel.\n        channel_id: The user-defined channel ID.\n        input_id: The user-defined input ID.\n        output_uri: Uri of the channel output folder in a Cloud Storage bucket.'
    client = LivestreamServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    input = f'projects/{project_id}/locations/{location}/inputs/{input_id}'
    name = f'projects/{project_id}/locations/{location}/channels/{channel_id}'
    channel = live_stream_v1.types.Channel(name=name, input_attachments=[live_stream_v1.types.InputAttachment(key='my-input', input=input)], output=live_stream_v1.types.Channel.Output(uri=output_uri), elementary_streams=[live_stream_v1.types.ElementaryStream(key='es_video', video_stream=live_stream_v1.types.VideoStream(h264=live_stream_v1.types.VideoStream.H264CodecSettings(profile='high', width_pixels=1280, height_pixels=720, bitrate_bps=3000000, frame_rate=30))), live_stream_v1.types.ElementaryStream(key='es_audio', audio_stream=live_stream_v1.types.AudioStream(codec='aac', channel_count=2, bitrate_bps=160000))], mux_streams=[live_stream_v1.types.MuxStream(key='mux_video', elementary_streams=['es_video'], segment_settings=live_stream_v1.types.SegmentSettings(segment_duration=duration.Duration(seconds=2))), live_stream_v1.types.MuxStream(key='mux_audio', elementary_streams=['es_audio'], segment_settings=live_stream_v1.types.SegmentSettings(segment_duration=duration.Duration(seconds=2)))], manifests=[live_stream_v1.types.Manifest(file_name='manifest.m3u8', type_='HLS', mux_streams=['mux_video', 'mux_audio'], max_segment_count=5)])
    operation = client.create_channel(parent=parent, channel=channel, channel_id=channel_id)
    response = operation.result(600)
    print(f'Channel: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location in which to create the channel.', default='us-central1')
    parser.add_argument('--channel_id', help='The user-defined channel ID.', required=True)
    parser.add_argument('--input_id', help='The user-defined input ID.', required=True)
    parser.add_argument('--output_uri', help='The Cloud Storage bucket (and optional folder) in which to save the livestream output.', required=True)
    args = parser.parse_args()
    create_channel(args.project_id, args.location, args.channel_id, args.input_id, args.output_uri)