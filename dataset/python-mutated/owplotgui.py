"""

.. index:: plot

######################################
GUI elements for plots (``owplotgui``)
######################################

This module contains functions and classes for creating GUI elements commonly used for plots.

.. autoclass:: OrientedWidget
    :show-inheritance:

.. autoclass:: StateButtonContainer
    :show-inheritance:

.. autoclass:: OWToolbar
    :show-inheritance:

.. autoclass:: OWButton
    :show-inheritance:

.. autoclass:: OrangeWidgets.plot.OWPlotGUI
    :members:

"""
import os
from AnyQt.QtWidgets import QWidget, QToolButton, QVBoxLayout, QHBoxLayout, QGridLayout, QMenu, QAction, QSizePolicy, QLabel, QStyledItemDelegate, QStyle, QListView
from AnyQt.QtGui import QIcon, QFont, QPalette
from AnyQt.QtCore import Qt, pyqtSignal, QSize, QRect, QPoint, QMimeData
from Orange.data import ContinuousVariable, DiscreteVariable
from Orange.widgets import gui
from Orange.widgets.gui import OrangeUserRole
from Orange.widgets.utils.listfilter import variables_filter
from Orange.widgets.utils.itemmodels import DomainModel, VariableListModel
from .owconstants import NOTHING, ZOOMING, SELECT, SELECT_POLYGON, PANNING, SELECTION_ADD, SELECTION_REMOVE, SELECTION_TOGGLE, SELECTION_REPLACE
__all__ = ['variables_selection', 'OrientedWidget', 'OWToolbar', 'StateButtonContainer', 'OWAction', 'OWButton', 'OWPlotGUI']
SIZE_POLICY_ADAPTING = (QSizePolicy.Expanding, QSizePolicy.Ignored)
SIZE_POLICY_FIXED = (QSizePolicy.Minimum, QSizePolicy.Maximum)

class VariableSelectionModel(VariableListModel):
    IsSelected = next(OrangeUserRole)
    SortRole = next(OrangeUserRole)
    selection_changed = pyqtSignal()

    def __init__(self, selected_vars, max_vars=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(enable_dnd=True)
        self.selected_vars = selected_vars
        self.max_vars = max_vars

    def is_selected(self, index):
        if False:
            return 10
        return self[index.row()] in self.selected_vars

    def is_full(self):
        if False:
            i = 10
            return i + 15
        if self.max_vars is None:
            return False
        else:
            return len(self.selected_vars) >= self.max_vars

    def data(self, index, role):
        if False:
            i = 10
            return i + 15
        if role == self.IsSelected:
            return self.is_selected(index)
        elif role == Qt.FontRole:
            font = QFont()
            font.setBold(self.is_selected(index))
            return font
        elif role == self.SortRole:
            if self.is_selected(index):
                return self.selected_vars.index(self[index.row()])
            else:
                return len(self.selected_vars) + index.row()
        else:
            return super().data(index, role)

    def toggle_item(self, index):
        if False:
            print('Hello World!')
        var = self[index.row()]
        if var in self.selected_vars:
            self.selected_vars.remove(var)
        elif not self.is_full():
            self.selected_vars.append(var)
        self.selection_changed.emit()

    def mimeData(self, indexlist):
        if False:
            while True:
                i = 10
        if len(indexlist) != 1:
            return None
        mime = QMimeData()
        mime.setData(self.MIME_TYPE, b'see properties: item_index')
        mime.setProperty('item_index', indexlist[0])
        return mime

    def dropMimeData(self, mime, action, row, column, parent):
        if False:
            for i in range(10):
                print('nop')
        if action == Qt.IgnoreAction:
            return True
        if not mime.hasFormat(self.MIME_TYPE):
            return False
        prev_index = mime.property('item_index')
        if prev_index is None:
            return False
        var = self[prev_index.row()]
        if self.is_selected(prev_index):
            self.selected_vars.remove(var)
        if row < len(self) and self.is_selected(self.index(row)):
            postpos = self.selected_vars.index(self[row])
            self.selected_vars.insert(postpos, var)
        elif row == 0 or self.is_selected(self.index(row - 1)):
            self.selected_vars.append(var)
        self.selection_changed.emit()
        return True

    def removeRows(self, *_):
        if False:
            for i in range(10):
                print('nop')
        return False

    def moveRows(self, *_):
        if False:
            print('Hello World!')
        return False

    def insertRows(self, *_):
        if False:
            return 10
        return False

class VariablesDelegate(QStyledItemDelegate):

    def paint(self, painter, option, index):
        if False:
            print('Hello World!')
        rect = QRect(option.rect)
        is_selected = index.data(VariableSelectionModel.IsSelected)
        full_selection = index.model().sourceModel().is_full()
        if option.state & QStyle.State_MouseOver:
            if not full_selection or (full_selection and is_selected):
                txt = [' Add ', ' Remove '][is_selected]
                txtw = painter.fontMetrics().horizontalAdvance(txt)
                painter.save()
                painter.setPen(Qt.NoPen)
                painter.setBrush(option.palette.brush(QPalette.Button))
                brect = QRect(rect.x() + rect.width() - 8 - txtw, rect.y(), txtw, rect.height())
                painter.drawRoundedRect(brect, 4, 4)
                painter.setPen(option.palette.color(QPalette.ButtonText))
                painter.drawText(brect, Qt.AlignCenter, txt)
                painter.restore()
        painter.save()
        double_pen = painter.pen()
        double_pen.setWidth(2 * double_pen.width())
        if is_selected:
            next = index.sibling(index.row() + 1, index.column())
            if not next.isValid():
                painter.setPen(double_pen)
                painter.drawLine(rect.bottomLeft(), rect.bottomRight())
            elif not next.data(VariableSelectionModel.IsSelected):
                painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        elif not index.row():
            down = QPoint(0, painter.pen().width())
            painter.setPen(double_pen)
            painter.drawLine(rect.topLeft() + down, rect.topRight() + down)
        else:
            prev = index.sibling(index.row() - 1, index.column())
            if prev.data(VariableSelectionModel.IsSelected):
                painter.drawLine(rect.topLeft(), rect.topRight())
        painter.restore()
        super().paint(painter, option, index)

class VariableSelectionView(QListView):

    def __init__(self, *args, acceptedType=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding))
        self.setMinimumHeight(10)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover)
        self.setSelectionMode(self.SingleSelection)
        self.setAutoScroll(False)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(self.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setUniformItemSizes(True)
        self.setItemDelegate(VariablesDelegate())
        self.setMinimumHeight(50)

    def sizeHint(self):
        if False:
            print('Hello World!')
        return QSize(1, 150)

    def mouseMoveEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().mouseMoveEvent(e)
        self.update()

    def leaveEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().leaveEvent(e)
        self.update()

    def startDrag(self, supportedActions):
        if False:
            for i in range(10):
                print('nop')
        super().startDrag(supportedActions)
        self.selectionModel().clearSelection()

def variables_selection(widget, master, model):
    if False:
        while True:
            i = 10

    def update_list():
        if False:
            return 10
        proxy.sort(0)
        proxy.invalidate()
        view.selectionModel().clearSelection()
    (filter_edit, view) = variables_filter(model=model, parent=master, view_type=VariableSelectionView)
    proxy = view.model()
    proxy.setSortRole(model.SortRole)
    model.selection_changed.connect(update_list)
    model.dataChanged.connect(update_list)
    model.modelReset.connect(update_list)
    model.rowsInserted.connect(update_list)
    view.clicked.connect(lambda index: model.toggle_item(proxy.mapToSource(index)))
    master.contextOpened.connect(update_list)
    widget.layout().addWidget(filter_edit)
    widget.layout().addSpacing(4)
    widget.layout().addWidget(view)

class OrientedWidget(QWidget):
    """
        A simple QWidget with a box layout that matches its ``orientation``.
    """

    def __init__(self, orientation, parent):
        if False:
            for i in range(10):
                print('nop')
        QWidget.__init__(self, parent)
        if orientation == Qt.Vertical:
            self._layout = QVBoxLayout()
        else:
            self._layout = QHBoxLayout()
        self.setLayout(self._layout)

class OWToolbar(OrientedWidget):
    """
        A toolbar is a container that can contain any number of buttons.

        :param gui: Used to create containers and buttons
        :type gui: :obj:`.OWPlotGUI`

        :param text: The name of this toolbar
        :type text: str

        :param orientation: The orientation of this toolbar, either Qt.Vertical or Qt.Horizontal
        :type tex: int

        :param buttons: A list of button identifiers to be added to this toolbar
        :type buttons: list of (int or tuple)

        :param parent: The toolbar's parent widget
        :type parent: :obj:`.QWidget`
    """

    def __init__(self, gui, text, orientation, buttons, parent, nomargin=False):
        if False:
            for i in range(10):
                print('nop')
        OrientedWidget.__init__(self, orientation, parent)
        self.buttons = {}
        self.groups = {}
        i = 0
        n = len(buttons)
        while i < n:
            if buttons[i] == gui.StateButtonsBegin:
                state_buttons = []
                for j in range(i + 1, n):
                    if buttons[j] == gui.StateButtonsEnd:
                        s = gui.state_buttons(orientation, state_buttons, self, nomargin)
                        self.buttons.update(s.buttons)
                        self.groups[buttons[i + 1]] = s
                        i = j
                        self.layout().addStretch()
                        break
                    else:
                        state_buttons.append(buttons[j])
            elif buttons[i] == gui.Spacing:
                self.layout().addSpacing(10)
            elif type(buttons[i] == int):
                self.buttons[buttons[i]] = gui.tool_button(buttons[i], self)
            elif len(buttons[i] == 4):
                gui.tool_button(buttons[i], self)
            else:
                self.buttons[buttons[i][0]] = gui.tool_button(buttons[i], self)
            i = i + 1

    def select_state(self, state):
        if False:
            while True:
                i = 10
        state_buttons = {NOTHING: 11, ZOOMING: 11, SELECT: 13, SELECT_POLYGON: 13, PANNING: 12}
        self.buttons[state_buttons[state]].click()

    def select_selection_behaviour(self, selection_behaviour):
        if False:
            print('Hello World!')
        self.buttons[13]._actions[21 + selection_behaviour].trigger()

class StateButtonContainer(OrientedWidget):
    """
        This class can contain any number of checkable buttons, of which only one can be selected
        at any time.

        :param gui: Used to create containers and buttons
        :type gui: :obj:`.OWPlotGUI`

        :param buttons: A list of button identifiers to be added to this toolbar
        :type buttons: list of (int or tuple)

        :param orientation: The orientation of this toolbar, either Qt.Vertical or Qt.Horizontal
        :type tex: int

        :param parent: The toolbar's parent widget
        :type parent: :obj:`.QWidget`
    """

    def __init__(self, gui, orientation, buttons, parent, nomargin=False):
        if False:
            while True:
                i = 10
        OrientedWidget.__init__(self, orientation, parent)
        self.buttons = {}
        if nomargin:
            self.layout().setContentsMargins(0, 0, 0, 0)
        self._clicked_button = None
        for i in buttons:
            b = gui.tool_button(i, self)
            b.triggered.connect(self.button_clicked)
            self.buttons[i] = b
            self.layout().addWidget(b)

    def button_clicked(self, checked):
        if False:
            while True:
                i = 10
        sender = self.sender()
        self._clicked_button = sender
        for button in self.buttons.values():
            button.setDown(button is sender)

    def button(self, id):
        if False:
            print('Hello World!')
        return self.buttons[id]

    def setEnabled(self, enabled):
        if False:
            i = 10
            return i + 15
        OrientedWidget.setEnabled(self, enabled)
        if enabled and self._clicked_button:
            self._clicked_button.click()

class OWAction(QAction):
    """
      A :obj:`QAction` with convenience methods for calling a callback or
      setting an attribute of the plot.
    """

    def __init__(self, plot, icon_name=None, attr_name='', attr_value=None, callback=None, parent=None):
        if False:
            print('Hello World!')
        QAction.__init__(self, parent)
        if type(callback) == str:
            callback = getattr(plot, callback, None)
        if callback:
            self.triggered.connect(callback)
        if attr_name:
            self._plot = plot
            self.attr_name = attr_name
            self.attr_value = attr_value
            self.triggered.connect(self.set_attribute)
        if icon_name:
            self.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../../icons', icon_name + '.png')))
            self.setIconVisibleInMenu(True)

    def set_attribute(self, clicked):
        if False:
            for i in range(10):
                print('nop')
        setattr(self._plot, self.attr_name, self.attr_value)

class OWButton(QToolButton):
    """
        A custom tool button which signal when its down state changes
    """
    downChanged = pyqtSignal(bool)

    def __init__(self, action=None, parent=None):
        if False:
            print('Hello World!')
        QToolButton.__init__(self, parent)
        self.setMinimumSize(30, 30)
        if action:
            self.setDefaultAction(action)

    def setDown(self, down):
        if False:
            for i in range(10):
                print('nop')
        if self.isDown() != down:
            self.downChanged[bool].emit(down)
        QToolButton.setDown(self, down)

class OWPlotGUI:
    """
        This class contains functions to create common user interface elements (QWidgets)
        for configuration and interaction with the ``plot``.

        It provides shorter versions of some methods in :obj:`.gui` that are directly related to an
        :obj:`.OWPlot` object.

        Normally, you don't have to construct this class manually. Instead, first create the plot,
        then use the :attr:`.OWPlot.gui` attribute.

        Most methods in this class have similar arguments, so they are explaned here in a single
        place.

        :param widget: The parent widget which will contain the newly created widget.
        :type widget: QWidget

        :param id: If ``id`` is an ``int``, a button is constructed from the default table.
                   Otherwise, ``id`` must be tuple with 5 or 6 elements. These elements
                   are explained in the next table.
        :type id: int or tuple

        :param ids: A list of widget identifiers
        :type ids: list of id

        :param text: The text displayed on the widget
        :type text: str

        When using widgets that are specific to your visualization and not included here, you have
        to provide your
        own widgets id's. They are a tuple with the following members:

        :param id: An optional unique identifier for the widget.
                   This is only needed if you want to retrive this widget using
                   :obj:`.OWToolbar.buttons`.
        :type id: int or str

        :param text: The text to be displayed on or next to the widget
        :type text: str

        :param attr_name: Name of attribute which will be set when the button is clicked.
                          If this widget is checkable, its check state will be set
                          according to the current value of this attribute.
                          If this parameter is empty or None, no attribute will be read or set.
        :type attr_name: str

        :param attr_value: The value that will be assigned to the ``attr_name`` when the button is
        clicked.
        :type attr: any

        :param callback: Function to be called when the button is clicked.
                         If a string is passed as ``callback``, a method by that name of ``plot``
                         will be called.
                         If this parameter is empty or ``None``, no function will be called
        :type callback: str or function

        :param icon_name: The filename of the icon for this widget, without the '.png' suffix.
        :type icon_name: str

    """
    JITTER_SIZES = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10]

    def __init__(self, master):
        if False:
            print('Hello World!')
        self._master = master
        self._plot = master.graph
        self.color_model = DomainModel(placeholder='(Same color)', valid_types=DomainModel.PRIMITIVE)
        self.shape_model = DomainModel(placeholder='(Same shape)', valid_types=DiscreteVariable)
        self.size_model = DomainModel(placeholder='(Same size)', valid_types=ContinuousVariable)
        self.label_model = DomainModel(placeholder='(No labels)')
        self.points_models = [self.color_model, self.shape_model, self.size_model, self.label_model]
    Spacing = 0
    ShowLegend = 2
    ShowFilledSymbols = 3
    ShowGridLines = 4
    PointSize = 5
    AlphaValue = 6
    Color = 7
    Shape = 8
    Size = 9
    Label = 10
    Zoom = 11
    Pan = 12
    Select = 13
    ZoomSelection = 15
    ZoomReset = 16
    ToolTipShowsAll = 17
    ClassDensity = 18
    RegressionLine = 19
    LabelOnlySelected = 20
    SelectionAdd = 21
    SelectionRemove = 22
    SelectionToggle = 23
    SelectionOne = 24
    SimpleSelect = 25
    SendSelection = 31
    ClearSelection = 32
    ShufflePoints = 33
    StateButtonsBegin = 35
    StateButtonsEnd = 36
    AnimatePlot = 41
    AnimatePoints = 42
    AntialiasPlot = 43
    AntialiasPoints = 44
    AntialiasLines = 45
    DisableAnimationsThreshold = 48
    AutoAdjustPerformance = 49
    JitterSizeSlider = 51
    JitterNumericValues = 52
    UserButton = 100
    default_zoom_select_buttons = [StateButtonsBegin, Zoom, Pan, Select, StateButtonsEnd, Spacing, SendSelection, ClearSelection]
    _buttons = {Zoom: ('Zoom', 'state', ZOOMING, None, 'Dlg_zoom'), ZoomReset: ('Reset zoom', None, None, None, 'Dlg_zoom_reset'), Pan: ('Pan', 'state', PANNING, None, 'Dlg_pan_hand'), SimpleSelect: ('Select', 'state', SELECT, None, 'Dlg_arrow'), Select: ('Select', 'state', SELECT, None, 'Dlg_arrow'), SelectionAdd: ('Add to selection', 'selection_behavior', SELECTION_ADD, None, 'Dlg_select_add'), SelectionRemove: ('Remove from selection', 'selection_behavior', SELECTION_REMOVE, None, 'Dlg_select_remove'), SelectionToggle: ('Toggle selection', 'selection_behavior', SELECTION_TOGGLE, None, 'Dlg_select_toggle'), SelectionOne: ('Replace selection', 'selection_behavior', SELECTION_REPLACE, None, 'Dlg_arrow'), SendSelection: ('Send selection', None, None, 'send_selection', 'Dlg_send'), ClearSelection: ('Clear selection', None, None, 'clear_selection', 'Dlg_clear'), ShufflePoints: ('ShufflePoints', None, None, 'shuffle_points', 'Dlg_sort')}
    _check_boxes = {AnimatePlot: ('Animate plot', 'animate_plot', 'update_animations'), AnimatePoints: ('Animate points', 'animate_points', 'update_animations'), AntialiasPlot: ('Antialias plot', 'antialias_plot', 'update_antialiasing'), AntialiasPoints: ('Antialias points', 'antialias_points', 'update_antialiasing'), AntialiasLines: ('Antialias lines', 'antialias_lines', 'update_antialiasing'), AutoAdjustPerformance: ('Disable effects for large datasets', 'auto_adjust_performance', 'update_performance')}
    '\n        The list of built-in buttons. It is a map of\n        id : (name, attr_name, attr_value, callback, icon_name)\n\n        .. seealso:: :meth:`.tool_button`\n    '

    def _get_callback(self, name, master=None):
        if False:
            return 10
        if type(name) == str:
            return getattr(master or self._plot, name)
        else:
            return name

    def _check_box(self, widget, value, label, cb_name, stateWhenDisabled=None):
        if False:
            i = 10
            return i + 15
        "\n            Adds a :obj:`.QCheckBox` to ``widget``.\n            When the checkbox is toggled, the attribute ``value`` of the plot object is set to\n            the checkbox' check state, and the callback ``cb_name`` is called.\n        "
        args = dict(master=self._plot, value=value, label=label, callback=self._get_callback(cb_name, self._plot), stateWhenDisabled=stateWhenDisabled)
        if isinstance(widget.layout(), QGridLayout):
            widget = widget.layout()
        if isinstance(widget, QGridLayout):
            checkbox = gui.checkBox(None, **args)
            widget.addWidget(checkbox, widget.rowCount(), 1)
            return checkbox
        else:
            return gui.checkBox(widget, **args)

    def antialiasing_check_box(self, widget):
        if False:
            i = 10
            return i + 15
        '\n            Creates a check box that toggles the Antialiasing of the plot\n        '
        self._check_box(widget, 'use_antialiasing', 'Use antialiasing', 'update_antialiasing')

    def jitter_size_slider(self, widget, label='Jittering: '):
        if False:
            for i in range(10):
                print('nop')
        return self.add_control(widget, gui.valueSlider, label, master=self._plot, value='jitter_size', values=getattr(self._plot, 'jitter_sizes', self.JITTER_SIZES), callback=self._plot.update_jittering)

    def jitter_numeric_check_box(self, widget):
        if False:
            print('Hello World!')
        self._check_box(widget=widget, value='jitter_continuous', label='Jitter numeric values', cb_name='update_jittering')

    def show_legend_check_box(self, widget):
        if False:
            for i in range(10):
                print('nop')
        '\n            Creates a check box that shows and hides the plot legend\n        '
        self._check_box(widget, 'show_legend', 'Show legend', 'update_legend_visibility')

    def tooltip_shows_all_check_box(self, widget):
        if False:
            print('Hello World!')
        gui.checkBox(widget=widget, master=self._master, value='tooltip_shows_all', label='Show all data on mouse hover')

    def class_density_check_box(self, widget):
        if False:
            for i in range(10):
                print('nop')
        self._master.cb_class_density = self._check_box(widget=widget, value='class_density', label='Show color regions', cb_name=self._plot.update_density, stateWhenDisabled=False)

    def regression_line_check_box(self, widget):
        if False:
            return 10
        self._master.cb_reg_line = self._check_box(widget=widget, value='show_reg_line', label='Show regression line', cb_name=self._plot.update_regression_line)

    def label_only_selected_check_box(self, widget):
        if False:
            while True:
                i = 10
        self._check_box(widget=widget, value='label_only_selected', label='Label only selection and subset', cb_name=self._plot.update_labels)

    def filled_symbols_check_box(self, widget):
        if False:
            while True:
                i = 10
        self._check_box(widget, 'show_filled_symbols', 'Show filled symbols', 'update_filled_symbols')

    def grid_lines_check_box(self, widget):
        if False:
            i = 10
            return i + 15
        self._check_box(widget, 'show_grid', 'Show gridlines', 'update_grid_visibility')

    def animations_check_box(self, widget):
        if False:
            for i in range(10):
                print('nop')
        '\n            Creates a check box that enabled or disables animations\n        '
        self._check_box(widget, 'use_animations', 'Use animations', 'update_animations')

    def add_control(self, widget, control, label, **args):
        if False:
            print('Hello World!')
        if isinstance(widget.layout(), QGridLayout):
            widget = widget.layout()
        if isinstance(widget, QGridLayout):
            row = widget.rowCount()
            element = control(None, **args)
            widget.addWidget(QLabel(label), row, 0)
            widget.addWidget(element, row, 1)
            return element
        else:
            return control(widget, label=label, **args)

    def _slider(self, widget, value, label, min_value, max_value, step, cb_name, show_number=False):
        if False:
            return 10
        return self.add_control(widget, gui.hSlider, label, master=self._plot, value=value, minValue=min_value, maxValue=max_value, step=step, createLabel=show_number, callback=self._get_callback(cb_name, self._master))

    def point_size_slider(self, widget, label='Symbol size: '):
        if False:
            return 10
        '\n            Creates a slider that controls point size\n        '
        return self._slider(widget, 'point_width', label, 1, 20, 1, 'sizes_changed')

    def alpha_value_slider(self, widget, label='Opacity: '):
        if False:
            i = 10
            return i + 15
        '\n            Creates a slider that controls point transparency\n        '
        return self._slider(widget, 'alpha_value', label, 0, 255, 10, 'colors_changed')

    def _combo(self, widget, value, label, cb_name, items=(), model=None):
        if False:
            return 10
        return self.add_control(widget, gui.comboBox, label, master=self._master, value=value, items=items, model=model, callback=self._get_callback(cb_name, self._master), orientation=Qt.Horizontal, sendSelectedValue=True, contentsLength=12, labelWidth=50, searchable=True)

    def color_value_combo(self, widget, label='Color: '):
        if False:
            i = 10
            return i + 15
        'Creates a combo box that controls point color'
        self._combo(widget, 'attr_color', label, 'colors_changed', model=self.color_model)

    def shape_value_combo(self, widget, label='Shape: '):
        if False:
            while True:
                i = 10
        'Creates a combo box that controls point shape'
        self._combo(widget, 'attr_shape', label, 'shapes_changed', model=self.shape_model)

    def size_value_combo(self, widget, label='Size: '):
        if False:
            return 10
        'Creates a combo box that controls point size'
        self._combo(widget, 'attr_size', label, 'sizes_changed', model=self.size_model)

    def label_value_combo(self, widget, label='Label: '):
        if False:
            for i in range(10):
                print('nop')
        'Creates a combo box that controls point label'
        self._combo(widget, 'attr_label', label, 'labels_changed', model=self.label_model)

    def box_spacing(self, widget):
        if False:
            i = 10
            return i + 15
        if isinstance(widget.layout(), QGridLayout):
            widget = widget.layout()
        if isinstance(widget, QGridLayout):
            space = QWidget()
            space.setFixedSize(12, 12)
            widget.addWidget(space, widget.rowCount(), 0)
        else:
            gui.separator(widget)

    def point_properties_box(self, widget, box='Attributes'):
        if False:
            while True:
                i = 10
        '\n            Creates a box with controls for common point properties.\n            Currently, these properties are point size and transparency.\n        '
        box = self.create_gridbox(widget, box)
        self.add_widgets([self.Color, self.Shape, self.Size, self.Label, self.LabelOnlySelected], box)
        return box

    def effects_box(self, widget, box=False):
        if False:
            print('Hello World!')
        '\n        Create a box with controls for common plot settings\n        '
        box = self.create_gridbox(widget, box)
        self.add_widgets([self.PointSize, self.AlphaValue, self.JitterSizeSlider], box)
        return box

    def plot_properties_box(self, widget, box=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a box with controls for common plot settings\n        '
        return self.create_box([self.ClassDensity, self.ShowLegend], widget, box, False)
    _functions = {ShowFilledSymbols: filled_symbols_check_box, JitterSizeSlider: jitter_size_slider, JitterNumericValues: jitter_numeric_check_box, ShowLegend: show_legend_check_box, ShowGridLines: grid_lines_check_box, ToolTipShowsAll: tooltip_shows_all_check_box, ClassDensity: class_density_check_box, RegressionLine: regression_line_check_box, LabelOnlySelected: label_only_selected_check_box, PointSize: point_size_slider, AlphaValue: alpha_value_slider, Color: color_value_combo, Shape: shape_value_combo, Size: size_value_combo, Label: label_value_combo, Spacing: box_spacing}

    def add_widget(self, id, widget):
        if False:
            return 10
        if id in self._functions:
            self._functions[id](self, widget)
        elif id in self._check_boxes:
            (label, attr, cb) = self._check_boxes[id]
            self._check_box(widget, attr, label, cb)

    def add_widgets(self, ids, widget):
        if False:
            i = 10
            return i + 15
        for id in ids:
            self.add_widget(id, widget)

    def create_box(self, ids, widget, box, name):
        if False:
            for i in range(10):
                print('nop')
        "\n            Creates a :obj:`.QGroupBox` with text ``name`` and adds it to ``widget``.\n            The ``ids`` argument is a list of widget ID's that will be added to this box\n        "
        if box is None:
            kwargs = {}
            box = gui.vBox(widget, name, margin=True, contentsMargins=(8, 4, 8, 4))
        self.add_widgets(ids, box)
        return box

    def create_gridbox(self, widget, box=True):
        if False:
            return 10
        grid = QGridLayout()
        grid.setColumnMinimumWidth(0, 50)
        grid.setColumnStretch(1, 1)
        b = gui.widgetBox(widget, box=box, orientation=grid)
        if not box:
            b.setContentsMargins(8, 4, 8, 4)
        grid.setVerticalSpacing(8)
        return b

    def _expand_id(self, id):
        if False:
            while True:
                i = 10
        if type(id) == int:
            (name, attr_name, attr_value, callback, icon_name) = self._buttons[id]
        elif len(id) == 4:
            (name, attr_name, attr_value, callback, icon_name) = id
            id = -1
        else:
            (id, name, attr_name, attr_value, callback, icon_name) = id
        return (id, name, attr_name, attr_value, callback, icon_name)

    def tool_button(self, id, widget):
        if False:
            i = 10
            return i + 15
        '\n            Creates an :obj:`.OWButton` and adds it to the parent ``widget``.\n        '
        (id, name, attr_name, attr_value, callback, icon_name) = self._expand_id(id)
        if id == OWPlotGUI.Select:
            b = self.menu_button(self.Select, [self.SelectionOne, self.SelectionAdd, self.SelectionRemove, self.SelectionToggle], widget)
        else:
            b = OWButton(parent=widget)
            ac = OWAction(self._plot, icon_name, attr_name, attr_value, callback, parent=b)
            b.setDefaultAction(ac)
        b.setToolTip(name)
        if widget.layout() is not None:
            widget.layout().addWidget(b)
        return b

    def menu_button(self, main_action_id, ids, widget):
        if False:
            i = 10
            return i + 15
        '\n            Creates an :obj:`.OWButton` with a popup-menu and adds it to the parent ``widget``.\n        '
        (id, _, attr_name, attr_value, callback, icon_name) = self._expand_id(main_action_id)
        b = OWButton(parent=widget)
        m = QMenu(b)
        b.setMenu(m)
        b._actions = {}
        m.triggered[QAction].connect(b.setDefaultAction)
        if main_action_id:
            main_action = OWAction(self._plot, icon_name, attr_name, attr_value, callback, parent=b)
            m.triggered.connect(main_action.trigger)
        for id in ids:
            (id, _, attr_name, attr_value, callback, icon_name) = self._expand_id(id)
            a = OWAction(self._plot, icon_name, attr_name, attr_value, callback, parent=m)
            m.addAction(a)
            b._actions[id] = a
        if m.actions():
            b.setDefaultAction(m.actions()[0])
        elif main_action_id:
            b.setDefaultAction(main_action)
        b.setPopupMode(QToolButton.MenuButtonPopup)
        b.setMinimumSize(40, 30)
        return b

    def state_buttons(self, orientation, buttons, widget, nomargin=False):
        if False:
            i = 10
            return i + 15
        '\n            This function creates a set of checkable buttons and connects them so that only one\n            may be checked at a time.\n        '
        c = StateButtonContainer(self, orientation, buttons, widget, nomargin)
        if widget.layout() is not None:
            widget.layout().addWidget(c)
        return c

    def toolbar(self, widget, text, orientation, buttons, nomargin=False):
        if False:
            return 10
        '\n            Creates an :obj:`.OWToolbar` with the specified ``text``, ``orientation``\n            and ``buttons`` and adds it to ``widget``.\n\n            .. seealso:: :obj:`.OWToolbar`\n        '
        t = OWToolbar(self, text, orientation, buttons, widget, nomargin)
        if nomargin:
            t.layout().setContentsMargins(0, 0, 0, 0)
        if widget.layout() is not None:
            widget.layout().addWidget(t)
        return t

    def zoom_select_toolbar(self, widget, text='Zoom / Select', orientation=Qt.Horizontal, buttons=default_zoom_select_buttons, nomargin=False):
        if False:
            for i in range(10):
                print('nop')
        t = self.toolbar(widget, text, orientation, buttons, nomargin)
        t.buttons[self.SimpleSelect].click()
        return t

    def theme_combo_box(self, widget):
        if False:
            i = 10
            return i + 15
        c = gui.comboBox(widget, self._plot, 'theme_name', 'Theme', callback=self._plot.update_theme, sendSelectedValue=1)
        c.addItem('Default')
        c.addItem('Light')
        c.addItem('Dark')
        return c

    def box_zoom_select(self, parent):
        if False:
            return 10
        box_zoom_select = gui.vBox(parent, 'Zoom/Select')
        zoom_select_toolbar = self.zoom_select_toolbar(box_zoom_select, nomargin=True, buttons=[self.StateButtonsBegin, self.SimpleSelect, self.Pan, self.Zoom, self.StateButtonsEnd, self.ZoomReset])
        buttons = zoom_select_toolbar.buttons
        buttons[self.Zoom].clicked.connect(self._plot.zoom_button_clicked)
        buttons[self.Pan].clicked.connect(self._plot.pan_button_clicked)
        buttons[self.SimpleSelect].clicked.connect(self._plot.select_button_clicked)
        buttons[self.ZoomReset].clicked.connect(self._plot.reset_button_clicked)
        return box_zoom_select