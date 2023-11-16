"""Google Cloud Transcoder sample for deleting a job template.

Example usage:
    python delete_job_template.py --project_id <project-id> --location <location> --template_id <template-id>
"""
import argparse
from google.cloud.video.transcoder_v1.services.transcoder_service import TranscoderServiceClient

def delete_job_template(project_id: str, location: str, template_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Deletes a job template.\n\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the template.\n        template_id: The user-defined template ID.'
    client = TranscoderServiceClient()
    name = f'projects/{project_id}/locations/{location}/jobTemplates/{template_id}'
    response = client.delete_job_template(name=name)
    print('Deleted job template')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the template.', required=True)
    parser.add_argument('--template_id', help='The job template ID.', required=True)
    args = parser.parse_args()
    delete_job_template(args.project_id, args.location, args.template_id)