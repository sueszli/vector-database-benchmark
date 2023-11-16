import grpc

class GreeterStub(object):
    """ Interface exported by the server. """

    def __init__(self, channel):
        if False:
            for i in range(10):
                print('nop')
        ' Constructor. \n    \n    Args: \n    channel: A grpc.Channel. \n    '
        self.SayHello = channel.unary_unary('/models.Greeter/SayHello')
        self.SayManyHellos = channel.unary_stream('/models.Greeter/SayManyHellos')

class GreeterServicer(object):
    """ Interface exported by the server. """

    def SayHello(self, request, context):
        if False:
            return 10
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SayManyHellos(self, request, context):
        if False:
            print('Hello World!')
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_GreeterServicer_to_server(servicer, server):
    if False:
        print('Hello World!')
    rpc_method_handlers = {'SayHello': grpc.unary_unary_rpc_method_handler(servicer.SayHello), 'SayManyHellos': grpc.unary_stream_rpc_method_handler(servicer.SayManyHellos)}
    generic_handler = grpc.method_handlers_generic_handler('models.Greeter', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))