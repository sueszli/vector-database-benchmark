"""The Python implementation of the GRPC helloworld.Greeter server."""
from concurrent import futures
import time
import grpc
import helloworld_pb2
import helloworld_pb2_grpc
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Greeter(helloworld_pb2_grpc.GreeterServicer):

    def SayHello(self, request, context):
        if False:
            return 10
        return helloworld_pb2.HelloReply(message='Hello, %s!' % request.name)

    def SayHelloAgain(self, request, context):
        if False:
            print('Hello World!')
        return helloworld_pb2.HelloReply(message='Hello again, %s!' % request.name)

def serve():
    if False:
        print('Hello World!')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(grace=0)
if __name__ == '__main__':
    serve()