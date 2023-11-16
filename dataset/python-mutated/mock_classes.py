import grpc
from grpc._server import _Context

class MockContext(_Context):

    def __init__(self) -> None:
        if False:
            return 10
        self.code: grpc.StatusCode = grpc.StatusCode.OK
        self.details: str = ''

    def set_details(self, details: str) -> None:
        if False:
            while True:
                i = 10
        self.details = details

    def set_code(self, code: grpc.StatusCode) -> None:
        if False:
            while True:
                i = 10
        self.code = code

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.code: grpc.StatusCode = grpc.StatusCode.OK
        self.details: str = ''