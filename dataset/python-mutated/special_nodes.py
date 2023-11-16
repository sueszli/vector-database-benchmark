import code
from contextlib import redirect_stdout, redirect_stderr
from packaging.version import Version
from ryvencore.addons.Logging import LoggingAddon
from ryven.node_env import *
guis = import_guis(__file__)

class NodeBase(Node):
    version = 'v0.2'
    GUI = guis.SpecialNodeGuiBase

    def have_gui(self):
        if False:
            while True:
                i = 10
        return hasattr(self, 'gui')

class DualNodeBase(NodeBase):
    """For nodes that can be active and passive"""
    GUI = guis.DualNodeBaseGui

    def __init__(self, params, active=True):
        if False:
            while True:
                i = 10
        super().__init__(params)
        self.active = active

    def make_passive(self):
        if False:
            for i in range(10):
                print('nop')
        self.delete_input(0)
        self.delete_output(0)
        self.active = False

    def make_active(self):
        if False:
            i = 10
            return i + 15
        self.create_input(type_='exec', insert=0)
        self.create_output(type_='exec', insert=0)
        self.active = True

    def get_state(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {'active': self.active}

    def set_state(self, data: dict, version):
        if False:
            i = 10
            return i + 15
        self.active = data['active']

class Checkpoint_Node(DualNodeBase):
    """Provides a simple checkpoint to reroute your connections"""
    title = 'checkpoint'
    init_inputs = [NodeInputType(type_='data')]
    init_outputs = [NodeOutputType(type_='data')]
    GUI = guis.CheckpointNodeGui

    def __init__(self, params):
        if False:
            print('Hello World!')
        super().__init__(params)
        self.active = False
    '\n    state transitions\n    '

    def clear_ports(self):
        if False:
            print('Hello World!')
        for i in range(len(self.outputs)):
            self.delete_output(0)
        for i in range(len(self.inputs)):
            self.delete_input(0)

    def make_active(self):
        if False:
            for i in range(10):
                print('nop')
        num_outputs = len(self.outputs)
        self.clear_ports()
        super().make_active()
        for i in range(1, num_outputs):
            self.add_output()

    def make_passive(self):
        if False:
            while True:
                i = 10
        num_outputs = len(self.outputs)
        super().make_passive()
        self.clear_ports()
        for i in range(num_outputs):
            self.add_output()

    def add_output(self):
        if False:
            return 10
        self.create_output(type_='exec' if self.active else 'data')

    def remove_output(self, index):
        if False:
            print('Hello World!')
        self.delete_output(index)
    '\n    update\n    '

    def update_event(self, inp=-1):
        if False:
            while True:
                i = 10
        if self.active and inp == 0:
            for i in range(len(self.outputs)):
                self.exec_output(i)
        elif not self.active:
            data = self.input(0)
            for i in range(len(self.outputs)):
                self.set_output_val(i, data)

class Button_Node(NodeBase):
    title = 'Button'
    init_inputs = []
    init_outputs = [NodeOutputType(type_='exec')]
    GUI = guis.ButtonNodeGui

    def update_event(self, inp=-1):
        if False:
            for i in range(10):
                print('nop')
        self.exec_output(0)

class Print_Node(DualNodeBase):
    title = 'Print'
    init_inputs = [NodeInputType(type_='exec'), NodeInputType()]
    init_outputs = [NodeOutputType(type_='exec')]

    def __init__(self, params):
        if False:
            return 10
        super().__init__(params, active=True)

    def update_event(self, inp=-1):
        if False:
            i = 10
            return i + 15
        if inp == 0:
            print(self.input(1 if self.active else 0).payload)
import logging

class Log_Node(DualNodeBase):
    title = 'Log'
    init_inputs = [NodeInputType(type_='exec'), NodeInputType('msg', type_='data')]
    init_outputs = [NodeOutputType(type_='exec')]
    GUI = guis.LogNodeGui
    logs = {}
    in_use = set()

    def __init__(self, params):
        if False:
            i = 10
            return i + 15
        super().__init__(params, active=True)
        self.number: int = None
        self.logger: logging.Logger = None

    def place_event(self):
        if False:
            i = 10
            return i + 15
        if self.number is None:
            self.number = len(self.logs)
            self.logs[self.number] = self.logs_ext().new_logger(self, 'Log Node')
            self.in_use.add(self.number)

    def logs_ext(self) -> LoggingAddon:
        if False:
            while True:
                i = 10
        return self.get_addon('Logging')

    def update_event(self, inp=-1):
        if False:
            while True:
                i = 10
        if inp == 0:
            msg = self.input(1 if self.active and inp == 0 else 0).payload
            self.logs[self.number].log(logging.INFO, msg=msg)

    def get_state(self) -> dict:
        if False:
            return 10
        return {**super().get_state(), 'number': self.number}

    def set_state(self, data: dict, version):
        if False:
            while True:
                i = 10
        if Version(version) < Version('0.2'):
            return
        super().set_state(data, version)
        n = data['number']
        if n not in self.in_use:
            self.number = n
        else:
            self.number = len(self.logs)
        self.logs[self.number] = self.logs_ext().new_logger(self, 'Log Node')

class Clock_Node(NodeBase):
    title = 'clock'
    init_inputs = [NodeInputType('delay'), NodeInputType('iterations')]
    init_outputs = [NodeOutputType(type_='exec')]
    GUI = guis.ClockNodeGui

    def __init__(self, params):
        if False:
            return 10
        super().__init__(params)
        self.running_with_qt = False

    def place_event(self):
        if False:
            while True:
                i = 10
        self.running_with_qt = self.GUI is not None
        if self.running_with_qt:
            from qtpy.QtCore import QTimer
            self.timer = QTimer()
            self.timer.timeout.connect(self.timeouted)
            self.iteration = 0

    def timeouted(self):
        if False:
            while True:
                i = 10
        self.exec_output(0)
        self.iteration += 1
        if -1 < self.input(1).payload <= self.iteration:
            self.stop()

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        if self.running_with_qt:
            self.timer.setInterval(self.input(0).payload)
            self.timer.start()
        else:
            import time
            for i in range(self.input(1).payload):
                self.exec_output(0)
                time.sleep(self.input(0).payload / 1000)

    def stop(self):
        if False:
            return 10
        assert self.running_with_qt
        self.timer.stop()
        self.iteration = 0

    def toggle(self):
        if False:
            i = 10
            return i + 15
        if self.running_with_qt:
            if self.timer.isActive():
                self.stop()
            else:
                self.start()

    def update_event(self, inp=-1):
        if False:
            return 10
        if self.running_with_qt:
            self.timer.setInterval(self.input(0).payload)

    def remove_event(self):
        if False:
            i = 10
            return i + 15
        if self.running_with_qt:
            self.stop()

class Slider_Node(NodeBase):
    title = 'slider'
    init_inputs = [NodeInputType('scl'), NodeInputType('round')]
    init_outputs = [NodeOutputType()]
    GUI = guis.SliderNodeGui

    def __init__(self, params):
        if False:
            print('Hello World!')
        super().__init__(params)
        self.val = 0

    def place_event(self):
        if False:
            for i in range(10):
                print('nop')
        self.update()

    def update_event(self, inp=-1):
        if False:
            return 10
        v = self.input(0).payload * self.val
        if self.input(1).payload:
            v = round(v)
        self.set_output_val(0, Data(v))

    def get_state(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {'val': self.val}

    def set_state(self, data: dict, version):
        if False:
            i = 10
            return i + 15
        self.val = data['val']

class _DynamicPorts_Node(NodeBase):
    init_inputs = []
    init_outputs = []
    GUI = guis.DynamicPortsGui

    def add_input(self):
        if False:
            i = 10
            return i + 15
        self.create_input()

    def remove_input(self, index):
        if False:
            while True:
                i = 10
        self.delete_input(index)

    def add_output(self):
        if False:
            return 10
        self.create_output()

    def remove_output(self, index):
        if False:
            print('Hello World!')
        self.delete_output(index)

class Exec_Node(_DynamicPorts_Node):
    title = 'exec'
    GUI = guis.ExecNodeGui

    def __init__(self, params):
        if False:
            print('Hello World!')
        super().__init__(params)
        self.code = None

    def update_event(self, inp=-1):
        if False:
            while True:
                i = 10
        exec(self.code)

    def get_state(self) -> dict:
        if False:
            while True:
                i = 10
        return {**super().get_state(), 'code': self.code}

    def set_state(self, data: dict, version):
        if False:
            for i in range(10):
                print('nop')
        super().set_state(data, version)
        self.code = data['code']

class Eval_Node(NodeBase):
    title = 'eval'
    init_inputs = []
    init_outputs = [NodeOutputType()]
    GUI = guis.EvalNodeGui

    def __init__(self, params):
        if False:
            print('Hello World!')
        super().__init__(params)
        self.number_param_inputs = 0
        self.expression_code = None

    def place_event(self):
        if False:
            return 10
        if self.number_param_inputs == 0:
            self.add_param_input()

    def add_param_input(self):
        if False:
            for i in range(10):
                print('nop')
        self.create_input()
        self.number_param_inputs += 1

    def remove_param_input(self, index):
        if False:
            i = 10
            return i + 15
        self.delete_input(index)
        self.number_param_inputs -= 1

    def update_event(self, inp=-1):
        if False:
            print('Hello World!')
        inp = [self.input(i).payload if self.input(i) is not None else None for i in range(self.number_param_inputs)]
        self.set_output_val(0, Data(eval(self.expression_code)))

    def get_state(self) -> dict:
        if False:
            i = 10
            return i + 15
        return {'num param inputs': self.number_param_inputs, 'expression code': self.expression_code}

    def set_state(self, data: dict, version):
        if False:
            while True:
                i = 10
        self.number_param_inputs = data['num param inputs']
        self.expression_code = data['expression code']

class Interpreter_Node(NodeBase):
    """
    Provides a python interpreter via a basic console with access to the
    node's properties.
    """
    title = 'interpreter'
    init_inputs = []
    init_outputs = []
    GUI = guis.InterpreterConsoleGui
    '\n    commands\n    '

    def clear(self):
        if False:
            print('Hello World!')
        self.hist.clear()
        self._hist_updated()

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.interp = code.InteractiveInterpreter(locals=locals())
    COMMANDS = {'clear': clear, 'reset': reset}
    '\n    behaviour\n    '

    def __init__(self, params):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(params)
        self.interp = None
        self.hist: [str] = []
        self.buffer: [str] = []
        self.reset()

    def _hist_updated(self):
        if False:
            i = 10
            return i + 15
        if self.have_gui():
            self.gui.main_widget().interp_updated()

    def process_input(self, cmds: str):
        if False:
            while True:
                i = 10
        m = self.COMMANDS.get(cmds)
        if m is not None:
            m()
        else:
            for l in cmds.splitlines():
                self.write(l)
                self.buffer.append(l)
            src = '\n'.join(self.buffer)

            def run_src():
                if False:
                    while True:
                        i = 10
                more_inp_required = self.interp.runsource(src, '<console>')
                if not more_inp_required:
                    self.buffer.clear()
            if self.session.gui:
                with redirect_stdout(self), redirect_stderr(self):
                    run_src()
            else:
                run_src()

    def write(self, line: str):
        if False:
            for i in range(10):
                print('nop')
        self.hist.append(line)
        self._hist_updated()

class Storage_Node(NodeBase):
    """
    Sequentially stores all the data provided at the input in an array.
    A shallow copy of the storage array is provided at the output
    """
    title = 'store'
    init_inputs = [NodeInputType()]
    init_outputs = [NodeOutputType()]
    GUI = guis.StorageNodeGui

    def __init__(self, params):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(params)
        self.storage = []

    def clear(self):
        if False:
            i = 10
            return i + 15
        self.storage.clear()
        self.set_output_val(0, Data([]))

    def update_event(self, inp=-1):
        if False:
            while True:
                i = 10
        self.storage.append(self.input(0).payload)
        self.set_output_val(0, Data(self.storage.copy()))

    def get_state(self) -> dict:
        if False:
            while True:
                i = 10
        return {'data': self.storage}

    def set_state(self, data: dict, version):
        if False:
            return 10
        self.storage = data['data']
import uuid

class LinkIN_Node(NodeBase):
    """
    Whenever a link IN node receives data (or an execution signal),
    if there is a linked LinkOUT node, it will receive the data
    and propagate it further.
    Notice that this breaks the data flow, which can have substantial
    performance implications and is generally not recommended.
    """
    title = 'link IN'
    init_inputs = [NodeInputType()]
    init_outputs = []
    GUI = guis.LinkIN_NodeGui
    INSTANCES = {}

    def __init__(self, params):
        if False:
            while True:
                i = 10
        super().__init__(params)
        self.ID: uuid.UUID = uuid.uuid4()
        self.INSTANCES[str(self.ID)] = self
        self.linked_node: LinkOUT_Node = None

    def add_input(self):
        if False:
            i = 10
            return i + 15
        self.create_input()
        if self.linked_node is not None:
            self.linked_node.add_output()

    def remove_input(self, index):
        if False:
            for i in range(10):
                print('nop')
        self.delete_input(index)
        if self.linked_node is not None:
            self.linked_node.remove_output(index)

    def update_event(self, inp=-1):
        if False:
            i = 10
            return i + 15
        if self.linked_node is not None:
            self.linked_node.set_output_val(inp, self.input(inp))

    def get_state(self) -> dict:
        if False:
            i = 10
            return i + 15
        return {'ID': str(self.ID)}

    def set_state(self, data: dict, version):
        if False:
            i = 10
            return i + 15
        if data['ID'] in self.INSTANCES:
            pass
        else:
            del self.INSTANCES[str(self.ID)]
            self.ID = uuid.UUID(data['ID'])
            self.INSTANCES[str(self.ID)] = self
            LinkOUT_Node.new_link_in_loaded(self)

    def remove_event(self):
        if False:
            print('Hello World!')
        if self.linked_node:
            self.linked_node.linked_node = None
            self.linked_node = None

class LinkOUT_Node(NodeBase):
    """The complement to the link IN node"""
    title = 'link OUT'
    init_inputs = []
    init_outputs = []
    GUI = guis.LinkOUT_NodeGui
    INSTANCES = []
    PENDING_LINK_BUILDS = {}

    @classmethod
    def new_link_in_loaded(cls, n: LinkIN_Node):
        if False:
            for i in range(10):
                print('nop')
        for (out_node, in_ID) in cls.PENDING_LINK_BUILDS.items():
            if in_ID == str(n.ID):
                out_node.link_to(n)

    def __init__(self, params):
        if False:
            while True:
                i = 10
        super().__init__(params)
        self.INSTANCES.append(self)
        self.linked_node: LinkIN_Node = None

    def link_to(self, n: LinkIN_Node):
        if False:
            print('Hello World!')
        self.linked_node = n
        n.linked_node = self
        o = len(self.outputs)
        i = len(self.linked_node.inputs)
        for j in range(i, o):
            self.delete_output(0)
        for j in range(o, i):
            self.create_output()
        self.update()

    def add_output(self):
        if False:
            while True:
                i = 10
        self.create_output()

    def remove_output(self, index):
        if False:
            print('Hello World!')
        self.delete_output(index)

    def update_event(self, inp=-1):
        if False:
            return 10
        if self.linked_node is None:
            return
        for i in range(len(self.outputs)):
            self.set_output_val(i, self.linked_node.input(i))

    def get_state(self) -> dict:
        if False:
            i = 10
            return i + 15
        if self.linked_node is None:
            return {}
        else:
            return {'linked ID': str(self.linked_node.ID)}

    def set_state(self, data: dict, version):
        if False:
            for i in range(10):
                print('nop')
        if len(data) > 0:
            n: LinkIN_Node = LinkIN_Node.INSTANCES.get(data['linked ID'])
            if n is None:
                self.PENDING_LINK_BUILDS[self] = data['linked ID']
            elif n.linked_node is None:
                n.linked_node = self
                self.linked_node = n

    def remove_event(self):
        if False:
            return 10
        if self.linked_node:
            self.linked_node.linked_node = None
            self.linked_node = None
nodes = [Checkpoint_Node, Button_Node, Print_Node, Log_Node, Clock_Node, Slider_Node, Exec_Node, Eval_Node, Storage_Node, LinkIN_Node, LinkOUT_Node, Interpreter_Node]