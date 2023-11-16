"""
This module contains the Scroll Flag panel
"""
import sys
from math import ceil
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QPainter, QColor, QCursor
from qtpy.QtWidgets import QStyle, QStyleOptionSlider, QApplication
from spyder.plugins.completion.api import DiagnosticSeverity
from spyder.plugins.editor.api.panel import Panel
from spyder.plugins.editor.utils.editor import is_block_safe
REFRESH_RATE = 1000
MAX_FLAGS = 1000

class ScrollFlagArea(Panel):
    """Source code editor's scroll flag area"""
    WIDTH = 24 if sys.platform == 'darwin' else 12
    FLAGS_DX = 4
    FLAGS_DY = 2

    def __init__(self):
        if False:
            print('Hello World!')
        Panel.__init__(self)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.scrollable = True
        self.setMouseTracking(True)
        self._unit_testing = False
        self._range_indicator_is_visible = False
        self._alt_key_is_down = False
        self._slider_range_color = QColor(Qt.gray)
        self._slider_range_color.setAlphaF(0.85)
        self._slider_range_brush = QColor(Qt.gray)
        self._slider_range_brush.setAlphaF(0.5)
        self._update_list_timer = QTimer(self)
        self._update_list_timer.setSingleShot(True)
        self._update_list_timer.timeout.connect(self.update_flags)
        self._dict_flag_list = {}

    def on_install(self, editor):
        if False:
            print('Hello World!')
        'Manages install setup of the pane.'
        super().on_install(editor)
        self._facecolors = {'warning': QColor(editor.warning_color), 'error': QColor(editor.error_color), 'todo': QColor(editor.todo_color), 'breakpoint': QColor(editor.breakpoint_color), 'occurrence': QColor(editor.occurrence_color), 'found_results': QColor(editor.found_results_color)}
        self._edgecolors = {key: color.darker(120) for (key, color) in self._facecolors.items()}
        editor.sig_focus_changed.connect(self.update)
        editor.sig_key_pressed.connect(self.keyPressEvent)
        editor.sig_key_released.connect(self.keyReleaseEvent)
        editor.sig_alt_left_mouse_pressed.connect(self.mousePressEvent)
        editor.sig_alt_mouse_moved.connect(self.mouseMoveEvent)
        editor.sig_leave_out.connect(self.update)
        editor.sig_flags_changed.connect(self.delayed_update_flags)
        editor.sig_theme_colors_changed.connect(self.update_flag_colors)

    @property
    def slider(self):
        if False:
            for i in range(10):
                print('nop')
        'This property holds whether the vertical scrollbar is visible.'
        return self.editor.verticalScrollBar().isVisible()

    def closeEvent(self, event):
        if False:
            while True:
                i = 10
        self._update_list_timer.stop()
        super().closeEvent(event)

    def sizeHint(self):
        if False:
            i = 10
            return i + 15
        'Override Qt method'
        return QSize(self.WIDTH, 0)

    def update_flag_colors(self, color_dict):
        if False:
            while True:
                i = 10
        '\n        Update the permanent Qt colors that are used for painting the flags\n        and the slider range with the new colors defined in the given dict.\n        '
        for (name, color) in color_dict.items():
            self._facecolors[name] = QColor(color)
            self._edgecolors[name] = self._facecolors[name].darker(120)

    def delayed_update_flags(self):
        if False:
            print('Hello World!')
        '\n        This function is called every time a flag is changed.\n        There is no need of updating the flags thousands of time by second,\n        as it is quite resources-heavy. This limits the calls to REFRESH_RATE.\n        '
        if self._update_list_timer.isActive():
            return
        self._update_list_timer.start(REFRESH_RATE)

    def update_flags(self):
        if False:
            i = 10
            return i + 15
        '\n        Update flags list.\n\n        This parses the entire file, which can take a lot of time for\n        large files. Save all the flags in lists for painting during\n        paint events.\n        '
        self._dict_flag_list = {'error': [], 'warning': [], 'todo': [], 'breakpoint': []}
        editor = self.editor
        block = editor.document().firstBlock()
        while block.isValid():
            data = block.userData()
            if data:
                if data.code_analysis:
                    for (_, _, severity, _) in data.code_analysis:
                        if severity == DiagnosticSeverity.ERROR:
                            flag_type = 'error'
                            break
                    else:
                        flag_type = 'warning'
                elif data.todo:
                    flag_type = 'todo'
                elif data.breakpoint:
                    flag_type = 'breakpoint'
                else:
                    flag_type = None
                if flag_type is not None:
                    self._dict_flag_list[flag_type].append(block)
            block = block.next()
        self.update()

    def paintEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override Qt method.\n        Painting the scroll flag area\n\n        There is two cases:\n            - The scroll bar is moving, in which case paint all flags.\n            - The scroll bar is not moving, only paint flags corresponding\n              to visible lines.\n        '
        groove_rect = self.get_scrollbar_groove_rect()
        scale_factor = groove_rect.height() / self.get_scrollbar_value_height()
        offset = groove_rect.y()
        rect_x = ceil(self.FLAGS_DX / 2)
        rect_w = self.WIDTH - self.FLAGS_DX
        rect_h = self.FLAGS_DY
        painter = QPainter(self)
        painter.fillRect(event.rect(), self.editor.sideareas_color)
        editor = self.editor
        last_line = editor.document().lastBlock().firstLineNumber()
        first_y_pos = self.value_to_position(0.5, scale_factor, offset) - self.FLAGS_DY / 2
        last_y_pos = self.value_to_position(last_line + 0.5, scale_factor, offset) - self.FLAGS_DY / 2
        line_height = last_y_pos - first_y_pos
        if line_height > 0:
            flag_height_lines = rect_h * last_line / line_height
        else:
            flag_height_lines = 0
        dict_flag_lists = {'occurrence': editor.occurrences, 'found_results': editor.found_results}
        dict_flag_lists.update(self._dict_flag_list)
        if sys.version_info[:2] > (3, 7):
            dict_flag_lists_iter = reversed(dict_flag_lists)
        else:
            dict_flag_lists_iter = dict_flag_lists
        for flag_type in dict_flag_lists_iter:
            painter.setBrush(self._facecolors[flag_type])
            painter.setPen(self._edgecolors[flag_type])
            if editor.verticalScrollBar().maximum() == 0:
                for block in dict_flag_lists[flag_type]:
                    if not is_block_safe(block):
                        continue
                    geometry = editor.blockBoundingGeometry(block)
                    rect_y = ceil(geometry.y() + geometry.height() / 2 + rect_h / 2)
                    painter.drawRect(rect_x, rect_y, rect_w, rect_h)
            elif last_line == 0:
                for block in dict_flag_lists[flag_type]:
                    if not is_block_safe(block):
                        continue
                    rect_y = ceil(first_y_pos)
                    painter.drawRect(rect_x, rect_y, rect_w, rect_h)
            elif len(dict_flag_lists[flag_type]) < MAX_FLAGS:
                next_line = 0
                for block in dict_flag_lists[flag_type]:
                    if not is_block_safe(block):
                        continue
                    block_line = block.firstLineNumber()
                    if block_line < next_line:
                        continue
                    next_line = block_line + flag_height_lines / 2
                    frac = block_line / last_line
                    rect_y = ceil(first_y_pos + frac * line_height)
                    painter.drawRect(rect_x, rect_y, rect_w, rect_h)
        if not self._unit_testing:
            alt = QApplication.queryKeyboardModifiers() & Qt.AltModifier
        else:
            alt = self._alt_key_is_down
        if self.slider:
            cursor_pos = self.mapFromGlobal(QCursor().pos())
            is_over_self = self.rect().contains(cursor_pos)
            is_over_editor = editor.rect().contains(editor.mapFromGlobal(QCursor().pos()))
            if is_over_self or (alt and is_over_editor):
                painter.setPen(self._slider_range_color)
                painter.setBrush(self._slider_range_brush)
                (x, y, width, height) = self.make_slider_range(cursor_pos, scale_factor, offset, groove_rect)
                painter.drawRect(x, y, width, height)
                self._range_indicator_is_visible = True
            else:
                self._range_indicator_is_visible = False

    def enterEvent(self, event):
        if False:
            while True:
                i = 10
        'Override Qt method'
        self.update()

    def leaveEvent(self, event):
        if False:
            return 10
        'Override Qt method'
        self.update()

    def mouseMoveEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Override Qt method'
        self.update()

    def mousePressEvent(self, event):
        if False:
            return 10
        'Override Qt method'
        if self.slider and event.button() == Qt.LeftButton:
            vsb = self.editor.verticalScrollBar()
            value = self.position_to_value(event.pos().y())
            vsb.setValue(int(value - vsb.pageStep() / 2))

    def keyReleaseEvent(self, event):
        if False:
            return 10
        'Override Qt method.'
        if event.key() == Qt.Key_Alt:
            self._alt_key_is_down = False
            self.update()

    def keyPressEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Override Qt method'
        if event.key() == Qt.Key_Alt:
            self._alt_key_is_down = True
            self.update()

    def get_vertical_offset(self):
        if False:
            while True:
                i = 10
        '\n        Return the vertical offset of the scroll flag area relative to the\n        top of the text editor.\n        '
        groove_rect = self.get_scrollbar_groove_rect()
        return groove_rect.y()

    def get_slider_min_height(self):
        if False:
            return 10
        "\n        Return the minimum height of the slider range based on that set for\n        the scroll bar's slider.\n        "
        return QApplication.instance().style().pixelMetric(QStyle.PM_ScrollBarSliderMin)

    def get_scrollbar_groove_rect(self):
        if False:
            return 10
        'Return the area in which the slider handle may move.'
        vsb = self.editor.verticalScrollBar()
        style = QApplication.instance().style()
        opt = QStyleOptionSlider()
        vsb.initStyleOption(opt)
        groove_rect = style.subControlRect(QStyle.CC_ScrollBar, opt, QStyle.SC_ScrollBarGroove, self)
        return groove_rect

    def get_scrollbar_position_height(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the pixel span height of the scrollbar area in which\n        the slider handle may move'
        groove_rect = self.get_scrollbar_groove_rect()
        return float(groove_rect.height())

    def get_scrollbar_value_height(self):
        if False:
            i = 10
            return i + 15
        'Return the value span height of the scrollbar'
        vsb = self.editor.verticalScrollBar()
        return vsb.maximum() - vsb.minimum() + vsb.pageStep()

    def get_scale_factor(self):
        if False:
            while True:
                i = 10
        "Return scrollbar's scale factor:\n        ratio between pixel span height and value span height"
        return self.get_scrollbar_position_height() / self.get_scrollbar_value_height()

    def value_to_position(self, y, scale_factor, offset):
        if False:
            for i in range(10):
                print('nop')
        'Convert value to position in pixels'
        vsb = self.editor.verticalScrollBar()
        return int((y - vsb.minimum()) * scale_factor + offset)

    def position_to_value(self, y):
        if False:
            return 10
        'Convert position in pixels to value'
        vsb = self.editor.verticalScrollBar()
        offset = self.get_vertical_offset()
        return vsb.minimum() + max([0, (y - offset) / self.get_scale_factor()])

    def make_slider_range(self, cursor_pos, scale_factor, offset, groove_rect):
        if False:
            print('Hello World!')
        '\n        Return the slider x and y positions and the slider width and height.\n        '
        vsb = self.editor.verticalScrollBar()
        slider_height = self.value_to_position(vsb.pageStep(), scale_factor, offset) - offset
        slider_height = max(slider_height, self.get_slider_min_height())
        min_ypos = offset
        max_ypos = groove_rect.height() + offset - slider_height
        slider_y = max(min_ypos, min(max_ypos, ceil(cursor_pos.y() - slider_height / 2)))
        return (1, slider_y, self.WIDTH - 2, slider_height)

    def wheelEvent(self, event):
        if False:
            print('Hello World!')
        'Override Qt method'
        self.editor.wheelEvent(event)

    def set_enabled(self, state):
        if False:
            while True:
                i = 10
        'Toggle scroll flag area visibility'
        self.enabled = state
        self.setVisible(state)