from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping
    from packaging.utils import NormalizedName
    from poetry.core.packages.package import Package

def get_extra_package_names(packages: Iterable[Package], extras: Mapping[NormalizedName, Iterable[NormalizedName]], extra_names: Collection[NormalizedName]) -> set[NormalizedName]:
    if False:
        print('Hello World!')
    '\n    Returns all package names required by the given extras.\n\n    :param packages: A collection of packages, such as from Repository.packages\n    :param extras: A mapping of `extras` names to lists of package names, as defined\n        in the `extras` section of `poetry.lock`.\n    :param extra_names: A list of strings specifying names of extra groups to resolve.\n    '
    from packaging.utils import canonicalize_name
    if not extra_names:
        return set()
    packages_by_name = {package.name: package for package in packages}
    seen_package_names = set()
    stack = [canonicalize_name(extra_package_name) for extra_name in extra_names for extra_package_name in extras.get(extra_name, ())]
    while stack:
        package_name = stack.pop()
        package = packages_by_name.get(package_name)
        if package is None or package.name in seen_package_names:
            continue
        seen_package_names.add(package.name)
        stack += [dependency.name for dependency in package.requires]
    return seen_package_names