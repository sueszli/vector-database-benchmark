import numpy as np
import threading
import logging
import warnings
import vaex.serialize
from .expression import FunctionSerializableJit
from . import expresso
logger = logging.getLogger('vaex.webserver.tornado')

class ExpressionStringMetal(expresso.ExpressionString):

    def pow(self, left, right):
        if False:
            i = 10
            return i + 15
        return 'pow({left}, {right})'.format(left=left, right=right)

def node_to_cpp(node, pretty=False):
    if False:
        while True:
            i = 10
    return ExpressionStringMetal(pretty=pretty).visit(node)
import faulthandler
faulthandler.enable()

@vaex.serialize.register
class FunctionSerializableMetal(FunctionSerializableJit):
    device = None

    def compile(self):
        if False:
            i = 10
            return i + 15
        try:
            import Metal
        except ImportError:
            logging.error('Failure to import Metal, please install pyobjc-framework-Metal')
            raise
        import objc
        dtype_out = vaex.dtype(self.return_dtype).numpy
        if dtype_out.name == 'float64':
            dtype_out = np.dtype('float32')
            warnings.warn('Casting output from float64 to float32 since Metal does not support float64')
        ast_node = expresso.parse_expression(self.expression)
        cppcode = node_to_cpp(ast_node)
        typemap = {'float32': 'float', 'float64': 'float'}
        for name in vaex.array_types._type_names_int:
            typemap[name] = f'{name}_t'
        typenames = [typemap[dtype.name] for dtype in self.argument_dtypes]
        metal_args = [f'const device {typename} *{name}_array [[buffer({i})]]' for (i, (typename, name)) in enumerate(zip(typenames, self.arguments))]
        code_get_scalar = [f'    {typename} {name} = {name}_array[id];\n' for (typename, name) in zip(typenames, self.arguments)]
        sourcecode = '\n#include <metal_stdlib>\nusing namespace metal;\n\nfloat arctan2(float y, float x) {\n    return atan2(y, x);\n}\n\ntemplate<typename T>\nT where(bool condition, T y, T x) {\n    return condition ? x : y;\n}\nkernel void vaex_kernel(%s,\n                        device %s *vaex_output [[buffer(%i)]],\n                        uint id [[thread_position_in_grid]]) {\n%s\n    vaex_output[id] = %s;\n}\n' % (', '.join(metal_args), typemap[dtype_out.name], len(metal_args), ''.join(code_get_scalar), cppcode)
        if self.verbose:
            print('Generated code:\n' + sourcecode)
        with open('test.metal', 'w') as f:
            print(f'Write to {f.name}')
            f.write(sourcecode)
        storage = threading.local()
        lock = threading.Lock()
        self.device = Metal.MTLCreateSystemDefaultDevice()
        opts = Metal.MTLCompileOptions.new()
        self.library = self.device.newLibraryWithSource_options_error_(sourcecode, opts, objc.NULL)
        if self.library[0] is None:
            msg = f'Error compiling: {sourcecode}, sourcecode'
            logger.error(msg)
            raise RuntimeError(msg)
        kernel_name = 'vaex_kernel'
        self.vaex_kernel = self.library[0].newFunctionWithName_(kernel_name)
        desc = Metal.MTLComputePipelineDescriptor.new()
        desc.setComputeFunction_(self.vaex_kernel)
        state = self.device.newComputePipelineStateWithDescriptor_error_(desc, objc.NULL)
        command_queue = self.device.newCommandQueue()

        def wrapper(*args):
            if False:
                return 10
            args = [vaex.array_types.to_numpy(ar) for ar in args]

            def getbuf(name, value=None, dtype=np.dtype('float32'), N=None):
                if False:
                    while True:
                        i = 10
                buf = getattr(storage, name, None)
                if value is not None:
                    N = len(value)
                    dtype = value.dtype
                if dtype.name == 'float64':
                    warnings.warn('Casting input argument from float64 to float32 since Metal does not support float64')
                    dtype = np.dtype('float32')
                nbytes = N * dtype.itemsize
                if buf is not None and buf.length() != nbytes:
                    buf = None
                if buf is None:
                    buf = self.device.newBufferWithLength_options_(nbytes, 0)
                    setattr(storage, name, buf)
                if value is not None:
                    mv = buf.contents().as_buffer(buf.length())
                    buf_as_numpy = np.frombuffer(mv, dtype=dtype)
                    buf_as_numpy[:] = value.astype(dtype, copy=False)
                return buf
            input_buffers = [getbuf(name, chunk) for (name, chunk) in zip(self.arguments, args)]
            output_buffer = getbuf('vaex_output', N=len(args[0]), dtype=dtype_out)
            buffers = input_buffers + [output_buffer]
            command_buffer = command_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(state)
            for (i, buf) in enumerate(buffers):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
            nitems = len(args[0])
            tpgrid = Metal.MTLSize(width=nitems, height=1, depth=1)
            tptgroup = Metal.MTLSize(width=state.threadExecutionWidth(), height=state.maxTotalThreadsPerThreadgroup() // state.threadExecutionWidth(), depth=1)
            encoder.dispatchThreads_threadsPerThreadgroup_(tpgrid, tptgroup)
            encoder.endEncoding()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            output_buffer_py = output_buffer.contents().as_buffer(output_buffer.length())
            result = np.frombuffer(output_buffer_py, dtype=dtype_out)
            return result
        return wrapper