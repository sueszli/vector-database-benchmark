"""Google Cloud Transcoder sample for creating a job that generates two spritesheets from the input video. Each spritesheet contains images that are captured periodically.

Example usage:
    python create_job_with_periodic_images_spritesheet.py --project_id <project-id> --location <location> --input_uri <uri> --output_uri <uri>
"""
import argparse
from google.cloud.video import transcoder_v1
from google.cloud.video.transcoder_v1.services.transcoder_service import TranscoderServiceClient
from google.protobuf import duration_pb2 as duration

def create_job_with_periodic_images_spritesheet(project_id: str, location: str, input_uri: str, output_uri: str) -> transcoder_v1.types.resources.Job:
    if False:
        return 10
    'Creates a job based on an ad-hoc job configuration that generates two spritesheets.\n\n    Args:\n        project_id: The GCP project ID.\n        location: The location to start the job in.\n        input_uri: Uri of the video in the Cloud Storage bucket.\n        output_uri: Uri of the video output folder in the Cloud Storage bucket.\n\n    Returns:\n        The job resource.\n    '
    client = TranscoderServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    job = transcoder_v1.types.Job()
    job.input_uri = input_uri
    job.output_uri = output_uri
    job.config = transcoder_v1.types.JobConfig(elementary_streams=[transcoder_v1.types.ElementaryStream(key='video-stream0', video_stream=transcoder_v1.types.VideoStream(h264=transcoder_v1.types.VideoStream.H264CodecSettings(height_pixels=360, width_pixels=640, bitrate_bps=550000, frame_rate=60))), transcoder_v1.types.ElementaryStream(key='audio-stream0', audio_stream=transcoder_v1.types.AudioStream(codec='aac', bitrate_bps=64000))], mux_streams=[transcoder_v1.types.MuxStream(key='sd', container='mp4', elementary_streams=['video-stream0', 'audio-stream0'])], sprite_sheets=[transcoder_v1.types.SpriteSheet(file_prefix='small-sprite-sheet', sprite_width_pixels=64, sprite_height_pixels=32, interval=duration.Duration(seconds=7)), transcoder_v1.types.SpriteSheet(file_prefix='large-sprite-sheet', sprite_width_pixels=128, sprite_height_pixels=72, interval=duration.Duration(seconds=7))])
    response = client.create_job(parent=parent, job=job)
    print(f'Job: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location to start this job in.', default='us-central1')
    parser.add_argument('--input_uri', help='Uri of the video in the Cloud Storage bucket.', required=True)
    parser.add_argument('--output_uri', help="Uri of the video output folder in the Cloud Storage bucket. Must end in '/'.", required=True)
    args = parser.parse_args()
    create_job_with_periodic_images_spritesheet(args.project_id, args.location, args.input_uri, args.output_uri)