from PyQt5.QtSvg import QGraphicsSvgItem, QSvgRenderer
from PyQt5.QtWidgets import QGraphicsScene, QWidget
from tribler.gui.sentry_mixin import AddBreadcrumbOnShowMixin
from tribler.gui.utilities import connect, get_image_path

def load_gears_animation():
    if False:
        for i in range(10):
            print('nop')
    svg_container = QGraphicsScene()
    svg_item = QGraphicsSvgItem()
    svg = QSvgRenderer(get_image_path('loading_animation.svg'))
    svg.repaintNeeded.connect(svg_item.update)
    svg_item.setSharedRenderer(svg)
    svg_container.addItem(svg_item)
    return svg_container
LOADING_ANIMATION = load_gears_animation()

class LoadingPage(AddBreadcrumbOnShowMixin, QWidget):
    """
    This page is presented when Tribler is starting.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        QWidget.__init__(self)
        self.loading_label = None
        self.upgrading = False

    def initialize_loading_page(self):
        if False:
            i = 10
            return i + 15
        self.window().loading_svg_view.setScene(LOADING_ANIMATION)
        connect(self.window().upgrade_manager.upgrader_tick, self.on_upgrader_tick)
        connect(self.window().upgrade_manager.upgrader_finished, self.upgrader_finished)
        connect(self.window().core_manager.events_manager.change_loading_text, self.change_loading_text)
        self.window().skip_conversion_btn.hide()
        self.window().force_shutdown_btn.hide()

    def upgrader_finished(self):
        if False:
            i = 10
            return i + 15
        self.window().skip_conversion_btn.hide()

    def on_upgrader_tick(self, text):
        if False:
            return 10
        if not self.upgrading:
            self.upgrading = True
            self.window().skip_conversion_btn.show()
        self.window().loading_text_label.setText(text)

    def change_loading_text(self, text):
        if False:
            while True:
                i = 10
        self.window().loading_text_label.setText(text)