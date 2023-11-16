from dataclasses import dataclass
from typing import Type, Optional
import inspect
from ryvencore import Node
from ryven.main.config import instance

def register_node_type(n: Type[Node]):
    if False:
        while True:
            i = 10
    '\n    Inspects and stores source code of a node type.\n    '
    has_gui = hasattr(n, 'GUI')
    has_mw = has_gui and n.GUI.main_widget_class is not None
    src = inspect.getsource(n)
    mw_src = inspect.getsource(n.GUI.main_widget_class) if has_mw else None
    inp_src = {name: inspect.getsource(cls) for (name, cls) in n.GUI.input_widget_classes.items()} if has_gui else None
    class_codes[n] = NodeTypeCodes(node_cls=src, main_widget_cls=mw_src, custom_input_widget_clss=inp_src)
    if instance.src_code_edits_enabled:
        mod_codes[n] = inspect.getsource(inspect.getmodule(n))
        if has_mw:
            mod_codes[n.GUI.main_widget_class] = inspect.getsource(inspect.getmodule(n.GUI.main_widget_class))
            for inp_cls in n.GUI.input_widget_classes.values():
                mod_codes[inp_cls] = inspect.getsource(inspect.getmodule(inp_cls))

@dataclass
class NodeTypeCodes:
    node_cls: str
    main_widget_cls: Optional[str]
    custom_input_widget_clss: {str: str}

class Inspectable:
    """
    Represents an object whose source code can be inspected.
    This is either a node or some node widget.
    Used by the code editor to store polymorphic references to
    objects which can be inspected.
    """

    def __init__(self, node, obj, code):
        if False:
            i = 10
            return i + 15
        self.node = node
        self.obj = obj
        self.code = code

class NodeInspectable(Inspectable):

    def __init__(self, node, code):
        if False:
            print('Hello World!')
        super().__init__(node, node, code)

class MainWidgetInspectable(Inspectable):
    pass

class CustomInputWidgetInspectable(Inspectable):
    pass
class_codes: {Type[Node]: {}} = {}
mod_codes: {Type: str} = {}
modif_codes: {object: str} = {}