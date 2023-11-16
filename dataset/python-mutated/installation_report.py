from typing import Any, Dict, Sequence
from pip._vendor.packaging.markers import default_environment
from pip import __version__
from pip._internal.req.req_install import InstallRequirement

class InstallationReport:

    def __init__(self, install_requirements: Sequence[InstallRequirement]):
        if False:
            print('Hello World!')
        self._install_requirements = install_requirements

    @classmethod
    def _install_req_to_dict(cls, ireq: InstallRequirement) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        assert ireq.download_info, f'No download_info for {ireq}'
        res = {'download_info': ireq.download_info.to_dict(), 'is_direct': ireq.is_direct, 'is_yanked': ireq.link.is_yanked if ireq.link else False, 'requested': ireq.user_supplied, 'metadata': ireq.get_dist().metadata_dict}
        if ireq.user_supplied and ireq.extras:
            res['requested_extras'] = sorted(ireq.extras)
        return res

    def to_dict(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {'version': '1', 'pip_version': __version__, 'install': [self._install_req_to_dict(ireq) for ireq in self._install_requirements], 'environment': default_environment()}