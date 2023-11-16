"""Google Cloud Video Stitcher sample for deleting a slate.
Example usage:
    python delete_slate.py --project_id <project-id> --location <location>         --slate_id <slate-id>
"""
import argparse
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient
from google.protobuf import empty_pb2 as empty

def delete_slate(project_id: str, location: str, slate_id: str) -> empty.Empty:
    if False:
        print('Hello World!')
    'Deletes a slate.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the slate.\n        slate_id: The user-defined slate ID.'
    client = VideoStitcherServiceClient()
    name = f'projects/{project_id}/locations/{location}/slates/{slate_id}'
    operation = client.delete_slate(name=name)
    response = operation.result()
    print('Deleted slate')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the slate.', required=True)
    parser.add_argument('--slate_id', help='The user-defined slate ID.', required=True)
    args = parser.parse_args()
    delete_slate(args.project_id, args.location, args.slate_id)