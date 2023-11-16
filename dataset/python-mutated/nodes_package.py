"""
This module, together with the node_env and gui_env defines Ryven's nodes
package system. It can be used outside of Ryven as well.
"""
import importlib.util
import os
import pathlib
from os.path import basename, dirname, splitext, normpath, join
from typing import Tuple, List, Type, Union, Set, Optional
from ryvencore import Node, Data
from ryven.main.utils import read_project, ryven_dir_path, abs_path_from_package_dir

class NodesPackage:
    """
    A small container to store meta data about imported node packages.
    """

    def __init__(self, directory: str):
        if False:
            for i in range(10):
                print('nop')
        self.name = basename(normpath(directory))
        self.directory = directory
        self.file_path = normpath(join(directory, 'nodes.py'))

    def __str__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}({self.name})'

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.__class__.__name__}({self.name}, {self.directory})'

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, NodesPackage):
            return self.name == other.name
        else:
            return self.name == str(other)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.name)

    def config_data(self):
        if False:
            for i in range(10):
                print('nop')
        return {'name': self.name, 'dir': self.directory}

def load_from_file(file: str=None, components_list: [str]=None) -> tuple:
    if False:
        for i in range(10):
            print('nop')
    '\n    Imports specified components from a python module with given file path.\n    '
    if components_list is None:
        components_list = []
    name = basename(file).split('.')[0]
    spec = importlib.util.spec_from_file_location(name, file)
    importlib.util.module_from_spec(spec)
    mod = spec.loader.load_module(name)
    comps = tuple([getattr(mod, c) for c in components_list])
    return comps

def import_nodes_package(package: NodesPackage=None, directory: str=None) -> Tuple[List[Type[Node]], List[Type[Data]]]:
    if False:
        while True:
            i = 10
    "Loads node and data classes from a Ryven nodes package and returns both in separate lists.\n\n    Can be used without a running Ryven instance, but you need to specify in which mode nodes should be loaded\n    by setting the environment variable RYVEN_MODE to either 'gui' (gui imports enabled) or 'no-gui'.\n\n    :param package: The NodesPackage object.\n    :param directory: The path to the directory where the nodes.py file is located, used if package is None.\n    :return: A tuple containing node types (classes) first, and the data types exported by the package second.\n    "
    if package is None:
        package = NodesPackage(directory)
    if 'RYVEN_MODE' not in os.environ:
        raise Exception("Please specify the environment variable RYVEN_MODE ('gui' or 'no-gui') before loading any packages. For example set os.environ['RYVEN_MODE'] = 'no-gui' for gui-less deployment.")
    from ryven import node_env
    node_env.NodesEnvRegistry.current_package = package
    load_from_file(package.file_path)
    node_types = node_env.NodesEnvRegistry.exported_nodes[-1]
    data_types = node_env.NodesEnvRegistry.exported_data_types[-1]
    return (node_types, data_types)

def process_nodes_packages(project_or_nodes: Union[Union[str, pathlib.Path], List[Union[str, pathlib.Path, NodesPackage]]], requested_packages: List[NodesPackage]=None) -> Tuple[Set[NodesPackage], List[pathlib.Path], Optional[dict]]:
    if False:
        i = 10
        return i + 15
    'Takes a project or list of node packages and additionally requested node\n    packages and checks whether the node packages are valid.\n\n    It also removes duplicates based on the name (and not the contents!).\n\n    :param project_or_nodes:\n        Either a path to a Ryven project or a list of node packages.\n        If a Ryven project is given, the required nodes packages specified\n        in the project file are looked for.\n        If a list is given, `NodesPackage` instances are  copied into the\n        resulting list; paths are considered to direct to \'nodes.py\'.\n        If \'nodes.py\' is found in the path,\n        a `NodesPackage` instance is created and added to the resulting list.\n        If \'nodes.py\' cannot be found in the path, the package is searched in\n        Ryven\'s example nodes dir, e.g. if "std" is given and not found\n        locally, the "std" package included in Ryven is loaded.\n    :param requested_packages:\n        A list of additional node package, which were requested. These take\n        precedence over `nodes`.\n        The default is `[]`.\n\n    :return:\n        A tuple of three elements:\n            - Set of available nodes required by the project or from list of nodes.\n            - Set of nodes required by the project or from list of nodes, which could not be found.\n            - Dictionary with the contents of the project or `None`.\n    '
    if requested_packages is None:
        requested_packages = []
    try:
        project_dict = read_project(project_or_nodes)
        node_pkg_paths = [p['dir'] for p in project_dict['required packages']]
    except TypeError:
        project_dict = None
        node_pkg_paths = project_or_nodes
    except KeyError:
        project_dict = None
        node_pkg_paths = []
    pkgs = set()
    pkgs_not_found = set()
    for pkg in node_pkg_paths:
        if isinstance(pkg, NodesPackage):
            pkgs.add(pkg)
        else:
            pkg_windows_path = pathlib.PureWindowsPath(pkg)
            pkg_posix_path = pathlib.PurePosixPath(pkg)
            if len(pkg_windows_path.parts) > len(pkg_posix_path.parts):
                pkg_path = pathlib.Path(pkg_windows_path)
            else:
                pkg_path = pathlib.Path(pkg_posix_path)
            if pkg_path.joinpath('nodes.py').exists():
                pkgs.add(NodesPackage(str(pkg_path)))
                continue
            pkg_custom_path = pathlib.Path(ryven_dir_path(), 'nodes', pkg)
            if pkg_custom_path.joinpath('nodes.py').exists():
                pkgs.add(NodesPackage(str(pkg_custom_path)))
                continue
            pkg_example_path = pathlib.Path(abs_path_from_package_dir('example_nodes'), pkg)
            if pkg_example_path.joinpath('nodes.py').exists():
                pkgs.add(NodesPackage(str(pkg_example_path)))
                continue
            pkgs_not_found.add(pkg_path)
    args_pkgs_names = [pkg.name for pkg in requested_packages]
    pkgs_not_found = [pkg_path for pkg_path in pkgs_not_found if pkg_path.name not in args_pkgs_names]
    return (pkgs, pkgs_not_found, project_dict)