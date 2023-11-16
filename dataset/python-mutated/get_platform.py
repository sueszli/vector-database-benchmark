import platform
ALL_PLATFORMS = {'x86_64', 'aarch64'}

def unify_aarch64(platform: str) -> str:
    if False:
        return 10
    return {'aarch64': 'aarch64', 'arm64': 'aarch64', 'x86_64': 'x86_64'}[platform]

def get_platform() -> str:
    if False:
        print('Hello World!')
    machine = platform.machine()
    return unify_aarch64(machine)