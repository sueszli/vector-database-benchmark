from qtpy.QtWidgets import QWidget, QPushButton, QHBoxLayout, QSlider, QColorDialog
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor

class FlowViewStylusModesWidget(QWidget):

    def __init__(self, flow_view):
        if False:
            print('Hello World!')
        super(FlowViewStylusModesWidget, self).__init__()
        self.setObjectName('FlowViewStylusModesWidget')
        self.flow_view = flow_view
        self.pen_color = QColor(255, 255, 0)
        self.stylus_buttons_visible = True
        self.stylus_button = QPushButton('stylus')
        self.stylus_button.clicked.connect(self.on_stylus_button_clicked)
        self.set_stylus_mode_comment_button = QPushButton('comment')
        self.set_stylus_mode_comment_button.clicked.connect(self.on_comment_button_clicked)
        self.set_stylus_mode_edit_button = QPushButton('edit')
        self.set_stylus_mode_edit_button.clicked.connect(self.on_edit_button_clicked)
        self.pen_color_button = QPushButton('color')
        self.pen_color_button.clicked.connect(self.on_choose_color_clicked)
        self.pen_width_slider = QSlider(Qt.Horizontal)
        self.pen_width_slider.setRange(1, 100)
        self.pen_width_slider.setValue(20)
        main_horizontal_layout = QHBoxLayout()
        main_horizontal_layout.addWidget(self.pen_color_button)
        main_horizontal_layout.addWidget(self.pen_width_slider)
        main_horizontal_layout.addWidget(self.set_stylus_mode_comment_button)
        main_horizontal_layout.addWidget(self.set_stylus_mode_edit_button)
        main_horizontal_layout.addWidget(self.stylus_button)
        self.setLayout(main_horizontal_layout)
        self.setStyleSheet('\n        QWidget#FlowViewStylusModesWidget {\n            background: transparent; \n        }\n                ')
        self.hide_stylus_buttons()
        self.hide_pen_style_widgets()

    def pen_width(self):
        if False:
            for i in range(10):
                print('nop')
        return self.pen_width_slider.value() / 20

    def hide_stylus_buttons(self):
        if False:
            print('Hello World!')
        self.set_stylus_mode_edit_button.hide()
        self.set_stylus_mode_comment_button.hide()
        self.stylus_buttons_visible = False

    def show_stylus_buttons(self):
        if False:
            return 10
        self.set_stylus_mode_edit_button.show()
        self.set_stylus_mode_comment_button.show()
        self.stylus_buttons_visible = True

    def hide_pen_style_widgets(self):
        if False:
            return 10
        self.pen_color_button.hide()
        self.pen_width_slider.hide()

    def show_pen_style_widgets(self):
        if False:
            print('Hello World!')
        self.pen_color_button.show()
        self.pen_width_slider.show()

    def on_stylus_button_clicked(self):
        if False:
            return 10
        if self.stylus_buttons_visible:
            self.hide_pen_style_widgets()
            self.hide_stylus_buttons()
        else:
            self.show_stylus_buttons()
        self.adjustSize()
        self.flow_view.set_stylus_proxy_pos()

    def on_edit_button_clicked(self):
        if False:
            while True:
                i = 10
        self.flow_view.stylus_mode = 'edit'
        self.hide_pen_style_widgets()
        self.hide_stylus_buttons()
        self.show_stylus_buttons()
        self.adjustSize()
        self.flow_view.set_stylus_proxy_pos()

    def on_comment_button_clicked(self):
        if False:
            i = 10
            return i + 15
        self.flow_view.stylus_mode = 'comment'
        self.show_pen_style_widgets()
        self.adjustSize()
        self.flow_view.set_stylus_proxy_pos()

    def on_choose_color_clicked(self):
        if False:
            print('Hello World!')
        self.pen_color = QColorDialog.getColor(self.pen_color, options=QColorDialog.ShowAlphaChannel, title='Choose pen color')
        self.update_color_button_SS()

    def update_color_button_SS(self):
        if False:
            while True:
                i = 10
        self.pen_color_button.setStyleSheet('\nQPushButton {\n    background-color: ' + self.pen_color.name() + ';\n}')

    def get_pen_settings(self):
        if False:
            i = 10
            return i + 15
        return {'color': self.pen_color.name(), 'base stroke weight': self.pen_width_slider.value() / 10}