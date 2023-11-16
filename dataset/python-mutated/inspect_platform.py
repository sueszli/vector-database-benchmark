import json
from dataclasses import asdict, dataclass
__all__ = ['Platform']

@dataclass(frozen=True)
class Platform:
    arch: str
    os: str
    distro: str
    distro_name: str
    distro_codename: str
    distro_like: str
    distro_major_version: str
    distro_version: str
    mac_release: str
    win_release: str

def _get_platform() -> Platform:
    if False:
        print('Hello World!')
    import distro
    import platform
    return Platform(arch=platform.machine(), os=platform.system(), distro=distro.id(), distro_name=distro.name(), distro_codename=distro.codename(), distro_like=distro.like(), distro_major_version=distro.major_version(), distro_version=distro.version(), mac_release=platform.mac_ver()[0], win_release=platform.win32_ver()[0])
if __name__ == '__main__':
    platform = _get_platform()
    platform_dict = asdict(platform)
    print(json.dumps(platform_dict))