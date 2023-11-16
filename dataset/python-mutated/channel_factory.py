"""Factory to create grpc channel."""
import grpc

class GRPCChannelFactory(grpc.StreamStreamClientInterceptor):
    DEFAULT_OPTIONS = [('grpc.keepalive_time_ms', 20000), ('grpc.keepalive_timeout_ms', 300000)]

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def insecure_channel(target, options=None):
        if False:
            while True:
                i = 10
        if options is None:
            options = []
        return grpc.insecure_channel(target, options=options + GRPCChannelFactory.DEFAULT_OPTIONS)

    @staticmethod
    def secure_channel(target, credentials, options=None):
        if False:
            print('Hello World!')
        if options is None:
            options = []
        return grpc.secure_channel(target, credentials, options=options + GRPCChannelFactory.DEFAULT_OPTIONS)