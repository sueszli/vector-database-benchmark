""" Tools for accessing distributions and resolving package names for them. """
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.Options import isExperimental
from nuitka.PythonFlavors import isAnacondaPython
from nuitka.PythonVersions import python_version
from .ModuleNames import ModuleName, checkModuleName
from .Utils import isLinux
_package_to_distribution = None

def getDistributionFiles(distribution):
    if False:
        while True:
            i = 10
    if hasattr(distribution, 'files'):
        for filename in distribution.files or ():
            filename = filename.as_posix()
            yield filename
    else:
        record_data = _getDistributionMetadataFileContents(distribution, 'RECORD')
        if record_data is not None:
            for line in record_data.splitlines():
                filename = line.split(',', 1)[0]
                yield filename

def _getDistributionMetadataFileContents(distribution, filename):
    if False:
        for i in range(10):
            print('nop')
    try:
        if hasattr(distribution, 'read_text'):
            result = distribution.read_text(filename)
        else:
            result = '\n'.join(distribution.get_metadata_lines(filename))
        return result
    except (FileNotFoundError, KeyError):
        return None

def getDistributionTopLevelPackageNames(distribution):
    if False:
        while True:
            i = 10
    'Returns the top level package names for a distribution.'
    top_level_txt = _getDistributionMetadataFileContents(distribution, 'top_level.txt')
    if top_level_txt is not None:
        result = [dirname.replace('/', '.') for dirname in top_level_txt.splitlines()]
    else:
        result = OrderedSet()
        for filename in getDistributionFiles(distribution):
            if filename.startswith('.'):
                continue
            first_path_element = filename.split('/')[0]
            if first_path_element.endswith('dist-info'):
                continue
            result.add(first_path_element)
    if not result:
        result = (getDistributionName(distribution),)
    return tuple(result)

def _pkg_resource_distributions():
    if False:
        i = 10
        return i + 15
    'Small replacement of distributions() of importlib.metadata that uses pkg_resources'
    from pip._vendor import pkg_resources
    return pkg_resources.working_set

def _initPackageToDistributionName():
    if False:
        return 10
    try:
        if isExperimental('force-pkg-resources-metadata'):
            raise ImportError
        try:
            from importlib.metadata import distributions
        except ImportError:
            from importlib_metadata import distributions
    except ImportError:
        distributions = _pkg_resource_distributions
    result = {}
    for distribution in distributions():
        for package_name in getDistributionTopLevelPackageNames(distribution):
            if not checkModuleName(package_name):
                continue
            package_name = ModuleName(package_name)
            if package_name not in result:
                result[package_name] = set()
            result[package_name].add(distribution)
    return result

def getDistributionsFromModuleName(module_name):
    if False:
        print('Hello World!')
    'Get the distribution names associated with a module name.\n\n    This can be more than one in case of namespace modules.\n    '
    global _package_to_distribution
    if _package_to_distribution is None:
        _package_to_distribution = _initPackageToDistributionName()
    while module_name not in _package_to_distribution and module_name.getPackageName() is not None:
        module_name = module_name.getPackageName()
    return tuple(sorted(_package_to_distribution.get(module_name, ()), key=getDistributionName))

def getDistributionFromModuleName(module_name):
    if False:
        print('Hello World!')
    'Get the distribution name associated with a module name.'
    distributions = getDistributionsFromModuleName(module_name)
    if not distributions:
        return None
    elif len(distributions) == 1:
        return distributions[0]
    else:
        return min(distributions, key=lambda dist: len(getDistributionName(dist)))

def getDistribution(distribution_name):
    if False:
        for i in range(10):
            print('nop')
    'Get a distribution by name.'
    try:
        if isExperimental('force-pkg-resources-metadata'):
            raise ImportError
        if python_version >= 896:
            from importlib import metadata
        else:
            import importlib_metadata as metadata
    except ImportError:
        from pip._vendor.pkg_resources import DistributionNotFound, get_distribution
        try:
            return get_distribution(distribution_name)
        except DistributionNotFound:
            return None
    else:
        try:
            return metadata.distribution(distribution_name)
        except metadata.PackageNotFoundError:
            return None
_distribution_to_installer = {}

def isDistributionCondaPackage(distribution_name):
    if False:
        print('Hello World!')
    if not isAnacondaPython():
        return False
    return getDistributionInstallerName(distribution_name) == 'conda'

def isDistributionPipPackage(distribution_name):
    if False:
        return 10
    return getDistributionInstallerName(distribution_name) == 'pip'

def isDistributionSystemPackage(distribution_name):
    if False:
        print('Hello World!')
    result = not isDistributionPipPackage(distribution_name) and (not isDistributionCondaPackage(distribution_name))
    if result:
        assert isLinux(), (distribution_name, getDistributionInstallerName(distribution_name))
    return result

def getDistributionInstallerName(distribution_name):
    if False:
        return 10
    'Get the installer name from a distribution object.\n\n    We might care of pip, anaconda, Debian, or whatever installed a\n    package.\n    '
    if distribution_name not in _distribution_to_installer:
        distribution = getDistribution(distribution_name)
        if distribution is None:
            if distribution_name == 'Pip':
                _distribution_to_installer[distribution_name] = 'default'
            else:
                _distribution_to_installer[distribution_name] = 'not_found'
        else:
            installer_name = _getDistributionMetadataFileContents(distribution, 'INSTALLER')
            if installer_name:
                _distribution_to_installer[distribution_name] = installer_name.strip().lower()
            elif hasattr(distribution, '_path'):
                distribution_path_parts = str(getattr(distribution, '_path')).split('/')
                if 'dist-packages' in distribution_path_parts and 'local' not in distribution_path_parts:
                    _distribution_to_installer[distribution_name] = 'Debian'
                else:
                    _distribution_to_installer[distribution_name] = 'Unknown'
            else:
                _distribution_to_installer[distribution_name] = 'Unknown'
    return _distribution_to_installer[distribution_name]

def getDistributionName(distribution):
    if False:
        print('Hello World!')
    'Get the distribution name from a distribution object.\n\n    We use importlib.metadata and pkg_resources version tuples interchangeable\n    and this is to abstract the difference is how to look up the name from\n    one.\n    '
    if hasattr(distribution, 'metadata'):
        return distribution.metadata['Name']
    else:
        return distribution.project_name

def getDistributionVersion(distribution):
    if False:
        for i in range(10):
            print('nop')
    'Get the distribution version string from a distribution object.\n\n    We use importlib.metadata and pkg_resources version tuples interchangeable\n    and this is to abstract the difference is how to look up the version from\n    one.\n    '
    if hasattr(distribution, 'metadata'):
        return distribution.metadata['Version']
    else:
        return distribution._version

def getDistributionLicense(distribution):
    if False:
        while True:
            i = 10
    'Get the distribution license from a distribution object.'
    license_name = distribution.metadata['License']
    if not license_name or license_name == 'UNKNOWN':
        for classifier in (value for (key, value) in distribution.metadata.items() if 'Classifier' in key):
            parts = [part.strip() for part in classifier.split('::')]
            if not parts:
                continue
            if parts[0] == 'License':
                license_name = parts[-1]
                break
    return license_name