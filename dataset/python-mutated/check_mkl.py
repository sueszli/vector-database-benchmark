from neon import logger as neon_logger
import os
import sys

def get_mkl_lib(device_id=None, verbose=False):
    if False:
        for i in range(10):
            print('nop')
    if sys.platform == 'win32':
        current_path = os.path.dirname(os.path.realpath(__file__))
        mkl_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'mklml.dll')
        if not os.path.isfile(mkl_engine_path):
            neon_logger.display('mklml.dll not found')
            return 0
        mkl_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'mklEngine.dll')
        if not os.path.isfile(mkl_engine_path):
            neon_logger.display('mklEngine.dll not found')
            return 0
        math_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'cmath.dll')
        if not os.path.isfile(math_engine_path):
            neon_logger.display('cmath.dll not found')
            return 0
        header_path = os.path.join(os.path.dirname(__file__), 'mklEngine', 'src', 'math_cpu.header')
        if os.path.isfile(header_path):
            neon_logger.display('math_cpu.header not found')
            return 0
        return 1
    elif sys.platform == 'darwin':
        current_path = os.path.dirname(os.path.realpath(__file__))
        mkl_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'mklEngine.dylib')
        if not os.path.isfile(mkl_engine_path):
            neon_logger.display('mklEngine.dylib not found')
            return 0
        math_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'cmath.dylib')
        if not os.path.isfile(math_engine_path):
            neon_logger.display('cmath.dylib not found')
            return 0
        header_path = os.path.join(os.path.dirname(__file__), 'mklEngine', 'src', 'math_cpu.header')
        if os.path.isfile(header_path):
            neon_logger.display('math_cpu.header not found')
            return 0
        return 1
    else:
        current_path = os.path.dirname(os.path.realpath(__file__))
        mkl_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'mklEngine.so')
        if not os.path.isfile(mkl_engine_path):
            neon_logger.display('mklEngine.so not found')
            return 0
        math_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'cmath.so')
        if not os.path.isfile(math_engine_path):
            neon_logger.display('cmath.so not found')
            return 0
        header_path = os.path.join(os.path.dirname(__file__), 'mklEngine', 'src', 'math_cpu.header')
        if os.path.isfile(header_path):
            neon_logger.display('math_cpu.header not found')
            return 0
        return 1