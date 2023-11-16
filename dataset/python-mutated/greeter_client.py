"""The Python implementation of the GRPC helloworld.Greeter client."""
import argparse
import grpc
import helloworld_pb2
import helloworld_pb2_grpc

def run(host, api_key):
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.insecure_channel(host)
    stub = helloworld_pb2_grpc.GreeterStub(channel)
    metadata = []
    if api_key:
        metadata.append(('x-api-key', api_key))
    response = stub.SayHello(helloworld_pb2.HelloRequest(name='you'), metadata=metadata)
    print('Greeter client received: ' + response.message)
    response = stub.SayHelloAgain(helloworld_pb2.HelloRequest(name='you'), metadata=metadata)
    print('Greeter client received: ' + response.message)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='localhost:50051', help='The server host.')
    parser.add_argument('--api_key', default=None, help='The API key to use for the call.')
    args = parser.parse_args()
    run(args.host, args.api_key)