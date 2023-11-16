from util import flatc, root_path
from pathlib import Path
grpc_examples_path = Path(root_path, 'grpc/examples')
greeter_schema = str(Path(grpc_examples_path, 'greeter.fbs'))
COMMON_ARGS = ['--grpc', '--bfbs-filenames', str(grpc_examples_path)]

def GenerateGRPCExamples():
    if False:
        return 10
    flatc(COMMON_ARGS + ['--go'], schema=greeter_schema, cwd=Path(grpc_examples_path, 'go/greeter'))
    flatc(COMMON_ARGS + ['--python'], schema=greeter_schema, cwd=Path(grpc_examples_path, 'python/greeter'))
    flatc(COMMON_ARGS + ['--swift', '--gen-json-emit'], schema=greeter_schema, cwd=Path(grpc_examples_path, 'swift/Greeter/Sources/Model'))
    flatc(COMMON_ARGS + ['--ts'], schema=greeter_schema, cwd=Path(grpc_examples_path, 'ts/greeter/src'))
if __name__ == '__main__':
    GenerateGRPCExamples()