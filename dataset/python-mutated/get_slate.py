"""Google Cloud Video Stitcher sample for getting a slate.
Example usage:
    python get_slate.py --project_id <project-id> --location <location>         --slate_id <slate-id>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient

def get_slate(project_id: str, location: str, slate_id: str) -> stitcher_v1.types.Slate:
    if False:
        for i in range(10):
            print('nop')
    'Gets a slate.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the slate.\n        slate_id: The user-defined slate ID.\n\n    Returns:\n        The slate resource.\n    '
    client = VideoStitcherServiceClient()
    name = f'projects/{project_id}/locations/{location}/slates/{slate_id}'
    response = client.get_slate(name=name)
    print(f'Slate: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the slate.', required=True)
    parser.add_argument('--slate_id', help='The user-defined slate ID.', required=True)
    args = parser.parse_args()
    get_slate(args.project_id, args.location, args.slate_id)