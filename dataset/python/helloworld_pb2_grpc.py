# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import helloworld_pb2 as helloworld__pb2


class GreeterStub:
    """The greeting service definition."""

    def __init__(self, channel):
        """Constructor.

        Args:
          channel: A grpc.Channel.
        """
        self.SayHello = channel.unary_unary(
            "/helloworld.Greeter/SayHello",
            request_serializer=helloworld__pb2.HelloRequest.SerializeToString,
            response_deserializer=helloworld__pb2.HelloReply.FromString,
        )
        self.SayHelloAgain = channel.unary_unary(
            "/helloworld.Greeter/SayHelloAgain",
            request_serializer=helloworld__pb2.HelloRequest.SerializeToString,
            response_deserializer=helloworld__pb2.HelloReply.FromString,
        )


class GreeterServicer:
    """The greeting service definition."""

    def SayHello(self, request, context):
        """Sends a greeting"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def SayHelloAgain(self, request, context):
        """Sends another greeting"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_GreeterServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "SayHello": grpc.unary_unary_rpc_method_handler(
            servicer.SayHello,
            request_deserializer=helloworld__pb2.HelloRequest.FromString,
            response_serializer=helloworld__pb2.HelloReply.SerializeToString,
        ),
        "SayHelloAgain": grpc.unary_unary_rpc_method_handler(
            servicer.SayHelloAgain,
            request_deserializer=helloworld__pb2.HelloRequest.FromString,
            response_serializer=helloworld__pb2.HelloReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "helloworld.Greeter", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
