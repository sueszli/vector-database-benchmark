"""Low level TF runtime client."""
from tensorflow.python import pywrap_tensorflow
from tensorflow.core.framework import function_pb2
from tensorflow.core.function.runtime_client import runtime_client_pybind
GlobalEagerContext = runtime_client_pybind.GlobalEagerContext
GlobalPythonEagerContext = runtime_client_pybind.GlobalPythonEagerContext

class Runtime(runtime_client_pybind.Runtime):

    def GetFunctionProto(self, name: str) -> function_pb2.FunctionDef:
        if False:
            return 10
        return function_pb2.FunctionDef.FromString(self.GetFunctionProtoString(name))

    def CreateFunction(self, function_def: function_pb2.FunctionDef):
        if False:
            return 10
        self.CreateFunctionFromString(function_def.SerializeToString())