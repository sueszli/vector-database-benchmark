import logging
from collections import OrderedDict
from typing import Dict, List
from pip._vendor.packaging.specifiers import LegacySpecifier
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import LegacyVersion
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.deprecation import deprecated
logger = logging.getLogger(__name__)

class RequirementSet:

    def __init__(self, check_supported_wheels: bool=True) -> None:
        if False:
            return 10
        'Create a RequirementSet.'
        self.requirements: Dict[str, InstallRequirement] = OrderedDict()
        self.check_supported_wheels = check_supported_wheels
        self.unnamed_requirements: List[InstallRequirement] = []

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        requirements = sorted((req for req in self.requirements.values() if not req.comes_from), key=lambda req: canonicalize_name(req.name or ''))
        return ' '.join((str(req.req) for req in requirements))

    def __repr__(self) -> str:
        if False:
            return 10
        requirements = sorted(self.requirements.values(), key=lambda req: canonicalize_name(req.name or ''))
        format_string = '<{classname} object; {count} requirement(s): {reqs}>'
        return format_string.format(classname=self.__class__.__name__, count=len(requirements), reqs=', '.join((str(req.req) for req in requirements)))

    def add_unnamed_requirement(self, install_req: InstallRequirement) -> None:
        if False:
            print('Hello World!')
        assert not install_req.name
        self.unnamed_requirements.append(install_req)

    def add_named_requirement(self, install_req: InstallRequirement) -> None:
        if False:
            print('Hello World!')
        assert install_req.name
        project_name = canonicalize_name(install_req.name)
        self.requirements[project_name] = install_req

    def has_requirement(self, name: str) -> bool:
        if False:
            print('Hello World!')
        project_name = canonicalize_name(name)
        return project_name in self.requirements and (not self.requirements[project_name].constraint)

    def get_requirement(self, name: str) -> InstallRequirement:
        if False:
            i = 10
            return i + 15
        project_name = canonicalize_name(name)
        if project_name in self.requirements:
            return self.requirements[project_name]
        raise KeyError(f'No project with the name {name!r}')

    @property
    def all_requirements(self) -> List[InstallRequirement]:
        if False:
            while True:
                i = 10
        return self.unnamed_requirements + list(self.requirements.values())

    @property
    def requirements_to_install(self) -> List[InstallRequirement]:
        if False:
            while True:
                i = 10
        'Return the list of requirements that need to be installed.\n\n        TODO remove this property together with the legacy resolver, since the new\n             resolver only returns requirements that need to be installed.\n        '
        return [install_req for install_req in self.all_requirements if not install_req.constraint and (not install_req.satisfied_by)]

    def warn_legacy_versions_and_specifiers(self) -> None:
        if False:
            while True:
                i = 10
        for req in self.requirements_to_install:
            version = req.get_dist().version
            if isinstance(version, LegacyVersion):
                deprecated(reason=f"pip has selected the non standard version {version} of {req}. In the future this version will be ignored as it isn't standard compliant.", replacement='set or update constraints to select another version or contact the package author to fix the version number', issue=12063, gone_in='24.0')
            for dep in req.get_dist().iter_dependencies():
                if any((isinstance(spec, LegacySpecifier) for spec in dep.specifier)):
                    deprecated(reason=f"pip has selected {req} {version} which has non standard dependency specifier {dep}. In the future this version of {req} will be ignored as it isn't standard compliant.", replacement='set or update constraints to select another version or contact the package author to fix the version number', issue=12063, gone_in='24.0')