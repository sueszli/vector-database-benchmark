"""Google Cloud Transcoder sample for getting the details for a job.

Example usage:
    python get_job.py --project_id <project-id> --location <location> --job_id <job-id>
"""
import argparse
from google.cloud.video import transcoder_v1
from google.cloud.video.transcoder_v1.services.transcoder_service import TranscoderServiceClient

def get_job(project_id: str, location: str, job_id: str) -> transcoder_v1.types.resources.Job:
    if False:
        for i in range(10):
            print('nop')
    'Gets a job.\n\n    Args:\n        project_id: The GCP project ID.\n        location: The location this job is in.\n        job_id: The job ID.\n\n    Returns:\n        The job resource.\n    '
    client = TranscoderServiceClient()
    name = f'projects/{project_id}/locations/{location}/jobs/{job_id}'
    response = client.get_job(name=name)
    print(f'Job: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the job.', required=True)
    parser.add_argument('--job_id', help='The job ID.', required=True)
    args = parser.parse_args()
    get_job(args.project_id, args.location, args.job_id)