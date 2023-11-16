"""Google Cloud Transcoder sample for creating a job template.

Example usage:
    python create_job_template.py --project_id <project-id> [--location <location>] [--template_id <template-id>]
"""
import argparse
from google.cloud.video import transcoder_v1
from google.cloud.video.transcoder_v1.services.transcoder_service import TranscoderServiceClient

def create_job_template(project_id: str, location: str, template_id: str) -> transcoder_v1.types.resources.JobTemplate:
    if False:
        while True:
            i = 10
    'Creates a job template.\n\n    Args:\n        project_id: The GCP project ID.\n        location: The location to store this template in.\n        template_id: The user-defined template ID.\n\n    Returns:\n        The job template resource.\n    '
    client = TranscoderServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    job_template = transcoder_v1.types.JobTemplate()
    job_template.name = f'projects/{project_id}/locations/{location}/jobTemplates/{template_id}'
    job_template.config = transcoder_v1.types.JobConfig(elementary_streams=[transcoder_v1.types.ElementaryStream(key='video-stream0', video_stream=transcoder_v1.types.VideoStream(h264=transcoder_v1.types.VideoStream.H264CodecSettings(height_pixels=360, width_pixels=640, bitrate_bps=550000, frame_rate=60))), transcoder_v1.types.ElementaryStream(key='video-stream1', video_stream=transcoder_v1.types.VideoStream(h264=transcoder_v1.types.VideoStream.H264CodecSettings(height_pixels=720, width_pixels=1280, bitrate_bps=2500000, frame_rate=60))), transcoder_v1.types.ElementaryStream(key='audio-stream0', audio_stream=transcoder_v1.types.AudioStream(codec='aac', bitrate_bps=64000))], mux_streams=[transcoder_v1.types.MuxStream(key='sd', container='mp4', elementary_streams=['video-stream0', 'audio-stream0']), transcoder_v1.types.MuxStream(key='hd', container='mp4', elementary_streams=['video-stream1', 'audio-stream0'])])
    response = client.create_job_template(parent=parent, job_template=job_template, job_template_id=template_id)
    print(f'Job template: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location to store this template in.', default='us-central1')
    parser.add_argument('--template_id', help='The job template ID.', default='my-job-template')
    args = parser.parse_args()
    create_job_template(args.project_id, args.location, args.template_id)