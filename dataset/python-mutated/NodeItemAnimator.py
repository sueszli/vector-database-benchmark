from qtpy.QtCore import QObject, QPropertyAnimation, Property
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QGraphicsItem

class NodeItemAnimator(QObject):

    def __init__(self, node_item):
        if False:
            while True:
                i = 10
        super(NodeItemAnimator, self).__init__()
        self.node_item = node_item
        self.animation_running = False
        self.title_activation_animation = QPropertyAnimation(self, b'p_title_color')
        self.title_activation_animation.setDuration(700)
        self.title_activation_animation.finished.connect(self.finished)
        self.body_activation_animation = QPropertyAnimation(self, b'p_body_color')
        self.body_activation_animation.setDuration(700)

    def start(self):
        if False:
            print('Hello World!')
        self.animation_running = True
        self.title_activation_animation.start()
        self.body_activation_animation.start()

    def stop(self):
        if False:
            return 10
        self.title_activation_animation.setCurrentTime(self.title_activation_animation.duration())
        self.body_activation_animation.setCurrentTime(self.body_activation_animation.duration())
        self.title_activation_animation.stop()
        self.body_activation_animation.stop()

    def finished(self):
        if False:
            return 10
        self.animation_running = False

    def running(self):
        if False:
            for i in range(10):
                print('nop')
        return self.animation_running

    def reload_values(self):
        if False:
            while True:
                i = 10
        self.stop()
        self.title_activation_animation.setKeyValueAt(0, self.get_title_color())
        self.title_activation_animation.setKeyValueAt(0.3, self.get_body_color().lighter().lighter())
        self.title_activation_animation.setKeyValueAt(1, self.get_title_color())
        self.body_activation_animation.setKeyValueAt(0, self.get_body_color())
        self.body_activation_animation.setKeyValueAt(0.3, self.get_body_color().lighter())
        self.body_activation_animation.setKeyValueAt(1, self.get_body_color())

    def fading_out(self):
        if False:
            print('Hello World!')
        return self.title_activation_animation.currentTime() / self.title_activation_animation.duration() >= 0.3

    def set_animation_max(self):
        if False:
            while True:
                i = 10
        self.title_activation_animation.setCurrentTime(0.3 * self.title_activation_animation.duration())
        self.body_activation_animation.setCurrentTime(0.3 * self.body_activation_animation.duration())

    def get_body_color(self):
        if False:
            for i in range(10):
                print('nop')
        return self.node_item.color

    def set_body_color(self, val):
        if False:
            return 10
        self.node_item.color = val
        QGraphicsItem.update(self.node_item)
    p_body_color = Property(QColor, get_body_color, set_body_color)

    def get_title_color(self):
        if False:
            for i in range(10):
                print('nop')
        return self.node_item.widget.title_label.color

    def set_title_color(self, val):
        if False:
            return 10
        self.node_item.widget.title_label.color = val
    p_title_color = Property(QColor, get_title_color, set_title_color)