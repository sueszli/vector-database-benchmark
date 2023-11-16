"""Google Cloud Live Stream sample for listing all inputs in a location.
Example usage:
    python list_inputs.py --project_id <project-id> --location <location>
"""
import argparse
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient, pagers

def list_inputs(project_id: str, location: str) -> pagers.ListInputsPager:
    if False:
        i = 10
        return i + 15
    'Lists all inputs in a location.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the inputs.'
    client = LivestreamServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    page_result = client.list_inputs(parent=parent)
    print('Inputs:')
    responses = []
    for response in page_result:
        print(response.name)
        responses.append(response)
    return responses
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the inputs.', required=True)
    args = parser.parse_args()
    list_inputs(args.project_id, args.location)