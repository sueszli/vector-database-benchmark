"""Google Cloud Live Stream sample for deleting an input.
Example usage:
    python delete_input.py --project_id <project-id> --location <location> --input_id <input-id>
"""
import argparse
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient
from google.protobuf import empty_pb2 as empty

def delete_input(project_id: str, location: str, input_id: str) -> empty.Empty:
    if False:
        while True:
            i = 10
    'Deletes an input.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the input.\n        input_id: The user-defined input ID.'
    client = LivestreamServiceClient()
    name = f'projects/{project_id}/locations/{location}/inputs/{input_id}'
    operation = client.delete_input(name=name)
    response = operation.result(600)
    print('Deleted input')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the input.', required=True)
    parser.add_argument('--input_id', help='The user-defined input ID.', required=True)
    args = parser.parse_args()
    delete_input(args.project_id, args.location, args.input_id)