"""A conformance test implementation for the Python protobuf library.

See conformance.proto for more information.
"""
import struct
import sys
import os
from google.protobuf import json_format
from google.protobuf import message
from google.protobuf import test_messages_proto3_pb2
import conformance_pb2
sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
sys.stdin = os.fdopen(sys.stdin.fileno(), 'rb', 0)
test_count = 0
verbose = False

class ProtocolError(Exception):
    pass

def do_test(request):
    if False:
        print('Hello World!')
    test_message = test_messages_proto3_pb2.TestAllTypes()
    response = conformance_pb2.ConformanceResponse()
    test_message = test_messages_proto3_pb2.TestAllTypes()
    try:
        if request.WhichOneof('payload') == 'protobuf_payload':
            try:
                test_message.ParseFromString(request.protobuf_payload)
            except message.DecodeError as e:
                response.parse_error = str(e)
                return response
        elif request.WhichOneof('payload') == 'json_payload':
            try:
                json_format.Parse(request.json_payload, test_message)
            except Exception as e:
                response.parse_error = str(e)
                return response
        else:
            raise ProtocolError("Request didn't have payload.")
        if request.requested_output_format == conformance_pb2.UNSPECIFIED:
            raise ProtocolError('Unspecified output format')
        elif request.requested_output_format == conformance_pb2.PROTOBUF:
            response.protobuf_payload = test_message.SerializeToString()
        elif request.requested_output_format == conformance_pb2.JSON:
            try:
                response.json_payload = json_format.MessageToJson(test_message)
            except Exception as e:
                response.serialize_error = str(e)
                return response
    except Exception as e:
        response.runtime_error = str(e)
    return response

def do_test_io():
    if False:
        i = 10
        return i + 15
    length_bytes = sys.stdin.read(4)
    if len(length_bytes) == 0:
        return False
    elif len(length_bytes) != 4:
        raise IOError('I/O error')
    length = struct.unpack('<I', length_bytes)[0]
    serialized_request = sys.stdin.read(length)
    if len(serialized_request) != length:
        raise IOError('I/O error')
    request = conformance_pb2.ConformanceRequest()
    request.ParseFromString(serialized_request)
    response = do_test(request)
    serialized_response = response.SerializeToString()
    sys.stdout.write(struct.pack('<I', len(serialized_response)))
    sys.stdout.write(serialized_response)
    sys.stdout.flush()
    if verbose:
        sys.stderr.write('conformance_python: request=%s, response=%s\n' % (request.ShortDebugString().c_str(), response.ShortDebugString().c_str()))
    global test_count
    test_count += 1
    return True
while True:
    if not do_test_io():
        sys.stderr.write('conformance_python: received EOF from test runner ' + 'after %s tests, exiting\n' % test_count)
        sys.exit(0)