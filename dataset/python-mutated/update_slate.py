"""Google Cloud Video Stitcher sample for updating a slate.
Example usage:
    python update_slate.py --project_id <project-id> --location <location>         --slate_id <slate-id> --slate_uri <uri>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient
from google.protobuf import field_mask_pb2 as field_mask

def update_slate(project_id: str, location: str, slate_id: str, slate_uri: str) -> stitcher_v1.types.Slate:
    if False:
        print('Hello World!')
    "Updates a slate.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the slate.\n        slate_id: The existing slate's ID.\n        slate_uri: Updated uri of the video slate; must be an MP4 video with at least one audio track.\n\n    Returns:\n        The slate resource.\n    "
    client = VideoStitcherServiceClient()
    name = f'projects/{project_id}/locations/{location}/slates/{slate_id}'
    slate = stitcher_v1.types.Slate(name=name, uri=slate_uri)
    update_mask = field_mask.FieldMask(paths=['uri'])
    operation = client.update_slate(slate=slate, update_mask=update_mask)
    response = operation.result()
    print(f'Updated slate: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the slate.', required=True)
    parser.add_argument('--slate_id', help="The existing slate's ID.", required=True)
    parser.add_argument('--slate_uri', help='Updated uri of the video slate; must be an MP4 video with at least one audio track.', required=True)
    args = parser.parse_args()
    update_slate(args.project_id, args.location, args.slate_id, args.slate_uri)