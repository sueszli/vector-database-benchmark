"""Google Cloud Video Stitcher sample for deleting a live config.
Example usage:
    python delete_live_config.py --project_id <project-id> --location <location>         --live_config_id <live-config-id>
"""
import argparse
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient
from google.protobuf import empty_pb2 as empty

def delete_live_config(project_id: str, location: str, live_config_id: str) -> empty.Empty:
    if False:
        while True:
            i = 10
    'Deletes a live config.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the live config.\n        live_config_id: The user-defined live config ID.'
    client = VideoStitcherServiceClient()
    name = f'projects/{project_id}/locations/{location}/liveConfigs/{live_config_id}'
    operation = client.delete_live_config(name=name)
    response = operation.result()
    print('Deleted live config')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the live config.', required=True)
    parser.add_argument('--live_config_id', help='The user-defined live config ID.', required=True)
    args = parser.parse_args()
    delete_live_config(args.project_id, args.location, args.live_config_id)