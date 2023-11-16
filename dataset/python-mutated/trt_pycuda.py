import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

def explicit_batch():
    if False:
        return 10
    return 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    if False:
        print('Hello World!')
    return val * 1 << 30

class HostDeviceMem(object):

    def __init__(self, host_mem, device_mem):
        if False:
            return 10
        "\n        Simple helper data class that's a little nicer to use than a 2-tuple.\n\n        Parameters\n        ----------\n        host_mem : host memory\n            Memory buffers of host\n        device_mem : device memory\n            Memory buffers of device\n        "
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Host:\n' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.__str__()

def allocate_buffers(engine):
    if False:
        print('Hello World!')
    '\n    Allocates all buffers required for an engine, i.e. host/device inputs/outputs.\n    NOTE: currently this function only supports NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag.\n\n    Parameters\n    ----------\n    engine : tensorrt.ICudaEngine\n        An ICudaEngine for executing inference on a built network\n\n    Returns\n    -------\n    list\n        All input HostDeviceMem of an engine\n    list\n        All output HostDeviceMem of an engine\n    GPU bindings\n        Device bindings\n    GPU stream\n        A stream is a sequence of commands (possibly issued by different host threads) that execute in order\n    '
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return (inputs, outputs, bindings, stream)

def do_inference_v2(context, bindings, inputs, outputs, stream):
    if False:
        i = 10
        return i + 15
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]