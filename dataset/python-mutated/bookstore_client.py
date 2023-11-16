"""The Python gRPC Bookstore Client Example."""
import argparse
from google.protobuf import empty_pb2
import grpc
import bookstore_pb2_grpc

def run(host, port, api_key, auth_token, timeout):
    if False:
        print('Hello World!')
    'Makes a basic ListShelves call against a gRPC Bookstore server.'
    channel = grpc.insecure_channel(f'{host}:{port}')
    stub = bookstore_pb2_grpc.BookstoreStub(channel)
    metadata = []
    if api_key:
        metadata.append(('x-api-key', api_key))
    if auth_token:
        metadata.append(('authorization', 'Bearer ' + auth_token))
    shelves = stub.ListShelves(empty_pb2.Empty(), timeout, metadata=metadata)
    print(f'ListShelves: {shelves}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='localhost', help='The host to connect to')
    parser.add_argument('--port', type=int, default=8000, help='The port to connect to')
    parser.add_argument('--timeout', type=int, default=10, help='The call timeout, in seconds')
    parser.add_argument('--api_key', default=None, help='The API key to use for the call')
    parser.add_argument('--auth_token', default=None, help='The JWT auth token to use for the call')
    args = parser.parse_args()
    run(args.host, args.port, args.api_key, args.auth_token, args.timeout)