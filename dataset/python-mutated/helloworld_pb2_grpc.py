import grpc
import helloworld_pb2 as helloworld__pb2

class GreeterStub:
    """The greeting service definition."""

    def __init__(self, channel):
        if False:
            print('Hello World!')
        'Constructor.\n\n        Args:\n          channel: A grpc.Channel.\n        '
        self.SayHello = channel.unary_unary('/helloworld.Greeter/SayHello', request_serializer=helloworld__pb2.HelloRequest.SerializeToString, response_deserializer=helloworld__pb2.HelloReply.FromString)
        self.SayHelloAgain = channel.unary_unary('/helloworld.Greeter/SayHelloAgain', request_serializer=helloworld__pb2.HelloRequest.SerializeToString, response_deserializer=helloworld__pb2.HelloReply.FromString)

class GreeterServicer:
    """The greeting service definition."""

    def SayHello(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        'Sends a greeting'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SayHelloAgain(self, request, context):
        if False:
            return 10
        'Sends another greeting'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_GreeterServicer_to_server(servicer, server):
    if False:
        while True:
            i = 10
    rpc_method_handlers = {'SayHello': grpc.unary_unary_rpc_method_handler(servicer.SayHello, request_deserializer=helloworld__pb2.HelloRequest.FromString, response_serializer=helloworld__pb2.HelloReply.SerializeToString), 'SayHelloAgain': grpc.unary_unary_rpc_method_handler(servicer.SayHelloAgain, request_deserializer=helloworld__pb2.HelloRequest.FromString, response_serializer=helloworld__pb2.HelloReply.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('helloworld.Greeter', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))