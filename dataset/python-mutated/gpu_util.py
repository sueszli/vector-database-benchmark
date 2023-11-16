"""Contains GPU utility functions."""
import collections
import re
_PHYSICAL_DEVICE_DESCRIPTION_REGEX = re.compile('name: ([^,]*), (?:.*compute capability: (\\d+)\\.(\\d+))?')
GpuInfo = collections.namedtuple('gpu_info', ['name', 'compute_capability'])

def compute_capability_from_device_desc(device_attrs):
    if False:
        while True:
            i = 10
    'Returns the GpuInfo given a DeviceAttributes proto.\n\n  Args:\n    device_attrs: A DeviceAttributes proto.\n\n  Returns\n    A gpu_info tuple. Both fields are None if `device_attrs` does not have a\n    valid physical_device_desc field.\n  '
    match = _PHYSICAL_DEVICE_DESCRIPTION_REGEX.search(device_attrs.physical_device_desc)
    if not match:
        return GpuInfo(None, None)
    cc = (int(match.group(2)), int(match.group(3))) if match.group(2) else None
    return GpuInfo(match.group(1), cc)