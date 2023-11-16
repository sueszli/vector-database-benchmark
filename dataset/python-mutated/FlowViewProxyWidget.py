from qtpy.QtWidgets import QGraphicsProxyWidget

class FlowViewProxyWidget(QGraphicsProxyWidget):
    """Ensures easy controls event handling for QProxyWidgets in the flow."""

    def __init__(self, flow_view, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super(FlowViewProxyWidget, self).__init__(parent)
        self.flow_view = flow_view

    def mousePressEvent(self, arg__1):
        if False:
            return 10
        QGraphicsProxyWidget.mousePressEvent(self, arg__1)
        if arg__1.isAccepted():
            self.flow_view.mouse_event_taken = True

    def mouseReleaseEvent(self, arg__1):
        if False:
            i = 10
            return i + 15
        self.flow_view.mouse_event_taken = True
        QGraphicsProxyWidget.mouseReleaseEvent(self, arg__1)

    def wheelEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        QGraphicsProxyWidget.wheelEvent(self, event)

    def keyPressEvent(self, arg__1):
        if False:
            return 10
        QGraphicsProxyWidget.keyPressEvent(self, arg__1)
        if arg__1.isAccepted():
            self.flow_view.ignore_key_event = True