from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QGraphicsTextItem
from urh.simulator.GraphicsItem import GraphicsItem
from urh.simulator.SimulatorCounterAction import SimulatorCounterAction
from urh.simulator.SimulatorGotoAction import SimulatorGotoAction
from urh.simulator.SimulatorItem import SimulatorItem
from urh.simulator.SimulatorSleepAction import SimulatorSleepAction
from urh.simulator.SimulatorTriggerCommandAction import SimulatorTriggerCommandAction

class ActionItem(GraphicsItem):

    def __init__(self, model_item: SimulatorItem, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(model_item=model_item, parent=parent)
        self.setFlag(QGraphicsTextItem.ItemIsPanel, True)
        self.text = QGraphicsTextItem(self)
        self.text.setFont(self.font)

    def update_flags(self):
        if False:
            for i in range(10):
                print('nop')
        if self.scene().mode == 0:
            self.set_flags(is_selectable=True, is_movable=True, accept_hover_events=True, accept_drops=True)

    def update_position(self, x_pos, y_pos):
        if False:
            while True:
                i = 10
        self.setPos(x_pos, y_pos)
        start_x = (self.scene().items_width() - self.labels_width()) / 2
        self.number.setPos(start_x, 0)
        start_x += self.number.boundingRect().width()
        self.text.setPos(start_x, 0)
        width = self.scene().items_width()
        self.prepareGeometryChange()
        self.bounding_rect = QRectF(0, 0, width, self.childrenBoundingRect().height() + 5)

    def labels_width(self):
        if False:
            i = 10
            return i + 15
        width = self.number.boundingRect().width()
        width += self.text.boundingRect().width()
        return width

class GotoActionItem(ActionItem):

    def __init__(self, model_item: SimulatorGotoAction, parent=None):
        if False:
            return 10
        assert isinstance(model_item, SimulatorGotoAction)
        super().__init__(model_item=model_item, parent=parent)

    def refresh(self):
        if False:
            for i in range(10):
                print('nop')
        text = '[Goto: '
        text += '...' if self.model_item.goto_target is None else self.model_item.goto_target
        text += ']'
        self.text.setPlainText(text)

class TriggerCommandActionItem(ActionItem):

    def __init__(self, model_item: SimulatorTriggerCommandAction, parent=None):
        if False:
            print('Hello World!')
        super().__init__(model_item=model_item, parent=parent)
        self.text.setPlainText('[Trigger command]')

class SleepActionItem(ActionItem):

    def __init__(self, model_item: SimulatorSleepAction, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(model_item=model_item, parent=parent)
        self.text.setPlainText('[' + model_item.caption + ']')

    def refresh(self):
        if False:
            print('Hello World!')
        self.text.setPlainText('[' + self.model_item.caption + ']')

class CounterActionItem(ActionItem):

    def __init__(self, model_item: SimulatorCounterAction, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(model_item=model_item, parent=parent)
        self.text.setPlainText('[Counter]')