"""Google Cloud Transcoder sample for creating a job based on a job preset.

Example usage:
    python create_job_from_preset.py --project_id <project-id> --location <location> --input_uri <uri> --output_uri <uri> [--preset <preset>]
"""
import argparse
from google.cloud.video import transcoder_v1
from google.cloud.video.transcoder_v1.services.transcoder_service import TranscoderServiceClient

def create_job_from_preset(project_id: str, location: str, input_uri: str, output_uri: str, preset: str) -> transcoder_v1.types.resources.Job:
    if False:
        while True:
            i = 10
    "Creates a job based on a job preset.\n\n    Args:\n        project_id: The GCP project ID.\n        location: The location to start the job in.\n        input_uri: Uri of the video in the Cloud Storage bucket.\n        output_uri: Uri of the video output folder in the Cloud Storage bucket.\n        preset: The preset template (for example, 'preset/web-hd').\n\n    Returns:\n        The job resource.\n    "
    client = TranscoderServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    job = transcoder_v1.types.Job()
    job.input_uri = input_uri
    job.output_uri = output_uri
    job.template_id = preset
    response = client.create_job(parent=parent, job=job)
    print(f'Job: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location to start this job in.', default='us-central1')
    parser.add_argument('--input_uri', help='Uri of the video in the Cloud Storage bucket.', required=True)
    parser.add_argument('--output_uri', help="Uri of the video output folder in the Cloud Storage bucket. Must end in '/'.", required=True)
    parser.add_argument('--preset', help="The preset template (for example, 'preset/web-hd').", default='preset/web-hd')
    args = parser.parse_args()
    create_job_from_preset(args.project_id, args.location, args.input_uri, args.output_uri, args.preset)