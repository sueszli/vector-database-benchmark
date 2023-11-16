import re
from typing import Dict, List
from apps.core import nvgpu
VENDOR = 'NVIDIA'
DEFAULT_DEVICES = ['all']
SPECIAL_DEVICES = {'void', 'none', 'all'}
DEVICE_INDEX_REGEX = re.compile('^(\\d+)$')
DEVICE_NAME_REGEX = re.compile('^(GPU-[a-fA-F0-9\\-]+)$')
DEFAULT_CAPABILITIES = ['compute', 'graphics', 'utility']
SPECIAL_CAPABILITIES = {'all'}
CAPABILITIES = {'compute', 'compat32', 'graphics', 'utility', 'video', 'display'}
DEFAULT_REQUIREMENTS: Dict[str, str] = dict()
REQUIREMENTS = {'cuda', 'driver', 'arch', 'brand'}

def is_supported() -> bool:
    if False:
        print('Hello World!')
    return nvgpu.is_supported()

def validate_devices(devices: List[str]) -> None:
    if False:
        return 10
    if not devices:
        raise ValueError(f'Missing {VENDOR} GPUs: {devices}')
    special_count = sum([d in SPECIAL_DEVICES for d in devices])
    has_mixed_devices = special_count > 0 and len(devices) > 1
    if special_count > 1 or has_mixed_devices:
        raise ValueError(f'Mixed {VENDOR} GPU devices: {devices}')
    if special_count > 0:
        return
    if all([DEVICE_INDEX_REGEX.match(d) for d in devices]):
        return
    if all([DEVICE_NAME_REGEX.match(d) for d in devices]):
        return
    raise ValueError(f'Invalid {VENDOR} GPU device names: {devices}')

def validate_capabilities(caps: List[str]) -> None:
    if False:
        print('Hello World!')
    if not caps:
        raise ValueError(f'Missing {VENDOR} GPU caps: {caps}')
    special_count = sum([c in SPECIAL_CAPABILITIES for c in caps])
    has_mixed_caps = special_count > 0 and len(caps) > 1
    if special_count > 1 or has_mixed_caps:
        raise ValueError(f'Mixed {VENDOR} GPU caps: {caps}')
    if special_count > 0:
        return
    if not all([c in CAPABILITIES for c in caps]):
        raise ValueError(f'Invalid {VENDOR} GPU caps: {caps}')

def validate_requirements(requirements: Dict[str, str]) -> None:
    if False:
        while True:
            i = 10
    ' Validate requirement names and check if a value was provided '
    for (name, val) in requirements.items():
        if name not in REQUIREMENTS:
            raise ValueError(f"Invalid {VENDOR} GPU requirement name: '{name}'")
        if not val:
            raise ValueError(f"Invalid {VENDOR} GPU requirement value: '{name}'='{val}'")