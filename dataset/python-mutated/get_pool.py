"""Google Cloud Live Stream sample for getting a pool.
Example usage:
    python get_pool.py --project_id <project-id> --location <location> --pool_id <pool-id>
"""
import argparse
from google.cloud.video import live_stream_v1
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient

def get_pool(project_id: str, location: str, pool_id: str) -> live_stream_v1.types.Pool:
    if False:
        for i in range(10):
            print('nop')
    'Gets a pool.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the pool.\n        pool_id: The user-defined pool ID.'
    client = LivestreamServiceClient()
    name = f'projects/{project_id}/locations/{location}/pools/{pool_id}'
    response = client.get_pool(name=name)
    print(f'Pool: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the pool.', required=True)
    parser.add_argument('--pool_id', help='The user-defined pool ID.', required=True)
    args = parser.parse_args()
    get_pool(args.project_id, args.location, args.pool_id)