"""
This module automatically imports all requirements for custom nodes.
"""
import os
from typing import Type
from ryven.main.packages.nodes_package import load_from_file, NodesPackage
from ryvencore import Node, NodeInputType, NodeOutputType, Data, serialize, deserialize

def init_node_env():
    if False:
        print('Hello World!')
    if os.environ['RYVEN_MODE'] == 'gui':
        import ryvencore_qt

def import_guis(origin_file: str, gui_file_name='gui.py'):
    if False:
        return 10
    '\n    Import all exported GUI classes from gui_file_name with respect to the origin_file location.\n    Returns an object with all exported gui classes as attributes for direct access.\n    '
    caller_location = os.path.dirname(origin_file)
    abs_path = os.path.join(caller_location, gui_file_name)
    if os.environ['RYVEN_MODE'] == 'gui':
        load_from_file(abs_path)
        from ryven import gui_env
        gui_classes_container = gui_env.GuiClassesRegistry.exported_guis[-1]
    else:

        class PlaceholderGuisContainer:

            def __getattr__(self, item):
                if False:
                    for i in range(10):
                        print('nop')
                return None
        gui_classes_container = PlaceholderGuisContainer()
    return gui_classes_container

class NodesEnvRegistry:
    """
    Statically stores custom `ryvencore.Node` and `ryvencore.Data` subclasses
    exported via export_nodes on import of a nodes package.
    After running the imported nodes.py module (which needs to call
    `export_nodes()` to run), Ryven can then retrieve the exported types from
    this class.
    """
    exported_nodes: [[Type[Node]]] = []
    exported_data_types: [[Type[Data]]] = []
    current_package: NodesPackage = None

def export_nodes(node_types: [Type[Node]], data_types: [Type[Data]]=None):
    if False:
        while True:
            i = 10
    '\n    Exports/exposes the specified nodes to Ryven for use in flows.\n    '
    if data_types is None:
        data_types = []
    for node_type in node_types:
        node_type.identifier_prefix = NodesEnvRegistry.current_package.name
        node_type.legacy_identifiers = [*node_type.legacy_identifiers, node_type.identifier if node_type.identifier else node_type.__name__]
    NodesEnvRegistry.exported_nodes.append(node_types)
    NodesEnvRegistry.exported_data_types.append(data_types)
    if os.environ['RYVEN_MODE'] == 'gui':
        from ryven.gui.code_editor.codes_storage import register_node_type
        for node_type in node_types:
            register_node_type(node_type)