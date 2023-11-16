"""Google Cloud Live Stream sample for updating a pool's peered network.
Example usage:
    python update_pool.py --project_id <project-id> --location <location>         --pool_id <pool-id> --peered_network <peered-network>
"""
import argparse
from google.cloud.video import live_stream_v1
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient
from google.protobuf import field_mask_pb2 as field_mask

def update_pool(project_id: str, location: str, pool_id: str, peered_network: str) -> live_stream_v1.types.Pool:
    if False:
        for i in range(10):
            print('nop')
    "Updates an pool.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the pool.\n        pool_id: The user-defined pool ID.\n        peered_network: The updated peer network (e.g.,\n        'projects/my-network-project-number/global/networks/my-network-name')."
    client = LivestreamServiceClient()
    name = f'projects/{project_id}/locations/{location}/pools/{pool_id}'
    pool = live_stream_v1.types.Pool(name=name, network_config=live_stream_v1.types.Pool.NetworkConfig(peered_network=peered_network))
    update_mask = field_mask.FieldMask(paths=['network_config'])
    operation = client.update_pool(pool=pool, update_mask=update_mask)
    response = operation.result()
    print(f'Updated pool: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the pool.', required=True)
    parser.add_argument('--pool_id', help='The user-defined pool ID.', required=True)
    parser.add_argument('--peered_network', help='The updated peer network.', required=True)
    args = parser.parse_args()
    update_pool(args.project_id, args.location, args.pool_id, args.peered_network)