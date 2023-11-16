from AnyQt.QtCore import Qt
from AnyQt.QtGui import QTransform
from AnyQt.QtWidgets import QGraphicsScene, QGraphicsSceneHelpEvent, QGraphicsView, QToolTip
__all__ = ['GraphicsScene', 'graphicsscene_help_event']

class GraphicsScene(QGraphicsScene):
    """
    A QGraphicsScene with better tool tip event dispatch.
    """

    def helpEvent(self, event: QGraphicsSceneHelpEvent) -> None:
        if False:
            while True:
                i = 10
        "\n        Reimplemented.\n\n        Send the help event to every graphics item that is under the event's\n        scene position (default `QGraphicsScene` only dispatches help events\n        to `QGraphicsProxyWidget`s.\n        "
        graphicsscene_help_event(self, event)

def graphicsscene_help_event(scene: QGraphicsScene, event: QGraphicsSceneHelpEvent) -> None:
    if False:
        return 10
    '\n    Send the help event to every graphics item that is under the `event`\n    scene position.\n    '
    widget = event.widget()
    if widget is not None and isinstance(widget.parentWidget(), QGraphicsView):
        view = widget.parentWidget()
        deviceTransform = view.viewportTransform()
    else:
        deviceTransform = QTransform()
    items = scene.items(event.scenePos(), Qt.IntersectsItemShape, Qt.DescendingOrder, deviceTransform)
    text = ''
    event.setAccepted(False)
    for item in items:
        scene.sendEvent(item, event)
        if event.isAccepted():
            return
        elif item.toolTip():
            text = item.toolTip()
            break
    QToolTip.showText(event.screenPos(), text, event.widget())
    event.setAccepted(bool(text))