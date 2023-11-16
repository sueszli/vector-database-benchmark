"""Google Cloud Transcoder sample for deleting a job.

Example usage:
    python delete_job.py --project_id <project-id> --location <location> --job_id <job-id>
"""
import argparse
from google.cloud.video.transcoder_v1.services.transcoder_service import TranscoderServiceClient

def delete_job(project_id: str, location: str, job_id: str) -> None:
    if False:
        return 10
    'Gets a job.\n\n    Args:\n        project_id: The GCP project ID.\n        location: The location this job is in.\n        job_id: The job ID.'
    client = TranscoderServiceClient()
    name = f'projects/{project_id}/locations/{location}/jobs/{job_id}'
    response = client.delete_job(name=name)
    print('Deleted job')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the job.', required=True)
    parser.add_argument('--job_id', help='The job ID.', required=True)
    args = parser.parse_args()
    delete_job(args.project_id, args.location, args.job_id)