from builtins import str
from neon import logger as neon_logger

def get_compute_capability(device_id=None, verbose=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Query compute capability through PyCuda and check it's 5.0 (Maxwell) or\n    greater.\n    5.0 (GTX750 Ti) only fp32 support\n    5.2 (GTX9xx series) required for fp16\n    By default, check all devices and return the highest compute capability.\n\n    Arguments:\n        device_id (int): CUDA device id. Default to None, will iterate over\n                         all devices if None.\n        verbose (bool): prints verbose logging if True, default False.\n\n    Returns:\n        float: Zero if no GPU is found, otherwise highest compute capability.\n    "
    try:
        import pycuda
        import pycuda.driver as drv
    except ImportError:
        if verbose:
            neon_logger.display('PyCUDA module not found')
        return 0
    try:
        drv.init()
    except pycuda._driver.RuntimeError as e:
        neon_logger.display('PyCUDA Runtime error: {0}'.format(str(e)))
        return 0
    major_string = pycuda._driver.device_attribute.COMPUTE_CAPABILITY_MAJOR
    minor_string = pycuda._driver.device_attribute.COMPUTE_CAPABILITY_MINOR
    full_version = []
    if device_id is None:
        device_id = list(range(drv.Device.count()))
    elif isinstance(device_id, int):
        device_id = [device_id]
    for i in device_id:
        major = drv.Device(i).get_attribute(major_string)
        minor = drv.Device(i).get_attribute(minor_string)
        full_version += [major + minor / 10.0]
    if verbose:
        neon_logger.display('Found GPU(s) with compute capability: {}'.format(full_version))
    return max(full_version)

def ensure_gpu_capability(device_id):
    if False:
        for i in range(10):
            print('nop')
    gpuflag = get_compute_capability(device_id) >= 3.0
    if gpuflag is False:
        raise RuntimeError('Device ' + str(device_id) + ' does not have CUDA compute ' + 'capability 3.0 or greater')

def get_device_count(verbose=False):
    if False:
        return 10
    '\n    Query device count through PyCuda.\n\n    Arguments:\n        verbose (bool): prints verbose logging if True, default False.\n\n    Returns:\n        int: Number of GPUs available.\n    '
    try:
        import pycuda
        import pycuda.driver as drv
    except ImportError:
        if verbose:
            neon_logger.display('PyCUDA module not found')
        return 0
    try:
        drv.init()
    except pycuda._driver.RuntimeError as e:
        neon_logger.display('PyCUDA Runtime error: {0}'.format(str(e)))
        return 0
    count = drv.Device.count()
    if verbose:
        neon_logger.display('Found {} GPU(s)'.format(count))
    return count
if __name__ == '__main__':
    neon_logger.display(get_compute_capability(verbose=False))