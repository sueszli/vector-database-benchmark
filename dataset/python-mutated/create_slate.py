"""Google Cloud Video Stitcher sample for creating a slate. A slate is displayed
when ads are not available.
Example usage:
    python create_slate.py --project_id <project-id> --location <location>         --slate_id <slate-id> --slate_uri <uri>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient

def create_slate(project_id: str, location: str, slate_id: str, slate_uri: str) -> stitcher_v1.types.Slate:
    if False:
        while True:
            i = 10
    'Creates a slate.\n    Args:\n        project_id: The GCP project ID.\n        location: The location in which to create the slate.\n        slate_id: The user-defined slate ID.\n        slate_uri: Uri of the video slate; must be an MP4 video with at least one audio track.\n\n    Returns:\n        The slate resource.\n    '
    client = VideoStitcherServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    slate = stitcher_v1.types.Slate(uri=slate_uri)
    operation = client.create_slate(parent=parent, slate_id=slate_id, slate=slate)
    response = operation.result()
    print(f'Slate: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location in which to create the slate.', default='us-central1')
    parser.add_argument('--slate_id', help='The user-defined slate ID.', required=True)
    parser.add_argument('--slate_uri', help='Uri of the video slate; must be an MP4 video with at least one audio track.', required=True)
    args = parser.parse_args()
    create_slate(args.project_id, args.location, args.slate_id, args.slate_uri)