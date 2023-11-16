import re
import html
from enum import IntEnum
from typing import Optional, Tuple, Union, Callable, NamedTuple, Dict, List
from AnyQt.QtCore import Qt, QTimer, QPoint, QItemSelectionModel, QSize, QAbstractListModel
from AnyQt.QtGui import QValidator, QPalette, QDoubleValidator, QIntValidator, QColor
from AnyQt.QtWidgets import QListView, QHBoxLayout, QStyledItemDelegate, QButtonGroup, QWidget, QLineEdit, QToolTip, QLabel, QApplication, QSpinBox, QSizePolicy, QRadioButton, QComboBox
from orangewidget.settings import Setting
from orangewidget.utils import listview
from Orange.data import Variable, ContinuousVariable, DiscreteVariable, TimeVariable, Domain, Table
import Orange.preprocess.discretize as disc
from Orange.widgets.utils.localization import pl
from Orange.widgets import widget, gui
from Orange.widgets.utils import unique_everseen
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output
from Orange.widgets.data.oweditdomain import FixedSizeButton
re_custom_sep = re.compile('\\s*,\\s*')
time_units = ['year', 'month', 'day', 'week', 'hour', 'minute', 'second']
INVALID_WIDTH = 'invalid width'
TOO_MANY_INTERVALS = 'too many intervals'

def _fixed_width_discretization(data: Table, var: Union[ContinuousVariable, str, int], width: str) -> Union[DiscreteVariable, str]:
    if False:
        print('Hello World!')
    '\n    Discretize numeric variable with fixed bin width. Used in method definition.\n\n    Width is given as string (coming from line edit). The labels for the new\n    variable will have the same number of digits; this is more appropriate\n    than the number of digits in the original variable, which may be too large.\n\n    Args:\n        data: data used to deduce the interval of values\n        var: variable to discretize\n        width: interval width\n\n    Returns:\n        Discrete variable, if successful; a string with error otherwise\n    '
    digits = len(width) - width.index('.') - 1 if '.' in width else 0
    try:
        width = float(width)
    except ValueError:
        return INVALID_WIDTH
    if width <= 0:
        return INVALID_WIDTH
    try:
        return disc.FixedWidth(width, digits)(data, var)
    except disc.TooManyIntervals:
        return TOO_MANY_INTERVALS

def _fixed_time_width_discretization(data: Table, var: Union[TimeVariable, str, int], width: str, unit: int) -> Union[DiscreteVariable]:
    if False:
        i = 10
        return i + 15
    '\n    Discretize time variable with fixed bin width. Used in method definition.\n\n    Width is given as string (coming from line edit).\n\n    Args:\n        data: data used to deduce the interval of values\n        var: variable to discretize\n        width: interval width\n        unit: 0 = year, 1 = month, 2 = week, 3 = day, 4 = hour, 5 = min, 6 = sec\n\n    Returns:\n        Discrete variable, if successful; a string with error otherwise\n    '
    try:
        width = int(width)
    except ValueError:
        return INVALID_WIDTH
    if width <= 0:
        return INVALID_WIDTH
    if unit == 3:
        width *= 7
    unit -= unit >= 3
    try:
        return disc.FixedTimeWidth(width, unit)(data, var)
    except disc.TooManyIntervals:
        return TOO_MANY_INTERVALS

def _mdl_discretization(data: Table, var: Union[ContinuousVariable, str, int]) -> Union[DiscreteVariable, str]:
    if False:
        while True:
            i = 10
    if not data.domain.has_discrete_class:
        return 'no discrete class'
    return disc.EntropyMDL()(data, var)

def _custom_discretization(_, var: Union[ContinuousVariable, str, int], points: str) -> Union[DiscreteVariable, str]:
    if False:
        while True:
            i = 10
    '\n    Discretize variable using custom thresholds. Used in method definition.\n\n    Thresholds are given as string (coming from line edit).\n\n    Args:\n        data: data used to deduce the interval of values\n        var: variable to discretize\n        points: thresholds\n\n    Returns:\n        Discrete variable, if successful; a string with error otherwise\n    '
    try:
        cuts = [float(x) for x in re_custom_sep.split(points.strip())]
    except ValueError:
        cuts = []
    if any((x >= y for (x, y) in zip(cuts, cuts[1:]))):
        cuts = []
    if not cuts:
        return 'invalid cuts'
    return disc.Discretizer.create_discretized_var(var, cuts)

class Methods(IntEnum):
    (Default, Keep, MDL, EqualFreq, EqualWidth, Remove, Custom, Binning, FixedWidth, FixedWidthTime) = range(10)

class MethodDesc(NamedTuple):
    """
    Definitions of all methods; used for creation of interface and calling
    """
    id_: Methods
    label: str
    short_desc: str
    tooltip: str
    function: Optional[Callable[..., Union[DiscreteVariable, str]]]
    controls: Tuple[str, ...] = ()
Options: Dict[Methods, MethodDesc] = {method.id_: method for method in (MethodDesc(Methods.Default, 'Use default setting', 'default', "Treat the variable as defined in 'default setting'", None, ()), MethodDesc(Methods.Keep, 'Keep numeric', 'keep', 'Keep the variable as is', lambda data, var: var, ()), MethodDesc(Methods.MDL, 'Entropy vs. MDL', 'entropy', 'Split values until MDL exceeds the entropy (Fayyad-Irani)\n(requires discrete class variable)', _mdl_discretization, ()), MethodDesc(Methods.EqualFreq, 'Equal frequency, intervals: ', 'equal freq, k={}', 'Create bins with same number of instances', lambda data, var, k: disc.EqualFreq(k)(data, var), ('freq_spin',)), MethodDesc(Methods.EqualWidth, 'Equal width, intervals: ', 'equal width, k={}', 'Create bins of the same width', lambda data, var, k: disc.EqualWidth(k)(data, var), ('width_spin',)), MethodDesc(Methods.Remove, 'Remove', 'remove', 'Remove variable', lambda *_: None, ()), MethodDesc(Methods.Binning, 'Natural binning, desired bins: ', 'binning, desired={}', 'Create bins with nice thresholds; try matching desired number of bins', lambda data, var, nbins: disc.Binning(nbins)(data, var), ('binning_spin',)), MethodDesc(Methods.FixedWidth, 'Fixed width: ', 'fixed width {}', 'Create bins with the given width (not for time variables)', _fixed_width_discretization, ('width_line',)), MethodDesc(Methods.FixedWidthTime, 'Time interval: ', 'time interval, {} {}', 'Create bins with the give width (for time variables)', _fixed_time_width_discretization, ('width_time_line', 'width_time_unit')), MethodDesc(Methods.Custom, 'Custom: ', 'custom: {}', 'Use manually specified thresholds', _custom_discretization, ('threshold_line',)))}

class VarHint(NamedTuple):
    """Description for settings"""
    method_id: Methods
    args: Tuple[Union[str, float, int]]

class DiscDesc(NamedTuple):
    """Data for list view model"""
    hint: VarHint
    points: str
    values: Tuple[str]
KeyType = Optional[Tuple[str, bool]]
DefaultHint = VarHint(Methods.Keep, ())
DefaultKey = None

def variable_key(var: ContinuousVariable) -> KeyType:
    if False:
        return 10
    'Key for that variable in var_hints and discretized_vars'
    return (var.name, isinstance(var, TimeVariable))

class ListViewSearch(listview.ListViewSearch):
    """
    A list view with two components shown above it:
    - a listview containing a single item representing default settings
    - a filter for search

    The class is based on listview.ListViewSearch and needs to have the same
    name in order to override its private method __layout.

    Inherited __init__ calls __layout, so `default_view` must be constructed
    there. Construction before calling super().__init__ doesn't work because
    PyQt does not allow it.
    """

    class DiscDelegate(QStyledItemDelegate):
        """
        A delegate that shows items (variables) with specific settings in bold
        """

        def initStyleOption(self, option, index):
            if False:
                i = 10
                return i + 15
            super().initStyleOption(option, index)
            option.font.setBold(index.data(Qt.UserRole).hint is not None)

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.default_view = None
        super().__init__(*args, preferred_size=QSize(350, -1), **kwargs)
        self.setItemDelegate(self.DiscDelegate(self))

    def select_default(self):
        if False:
            i = 10
            return i + 15
        'Select the item representing default settings'
        index = self.default_view.model().index(0)
        self.default_view.selectionModel().select(index, QItemSelectionModel.Select)

    def __layout(self):
        if False:
            i = 10
            return i + 15
        if self.default_view is None:
            view = self.default_view = QListView(self)
            view.setModel(DefaultDiscModel())
            view.verticalScrollBar().setDisabled(True)
            view.horizontalScrollBar().setDisabled(True)
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            font = view.font()
            font.setBold(True)
            view.setFont(font)
        else:
            view = self.default_view
        margins = self.viewportMargins()
        def_height = view.sizeHintForRow(0) + 2 * view.spacing() + 2
        view.setGeometry(0, 0, self.geometry().width(), def_height)
        view.setFixedHeight(def_height)
        search = self.__search
        src_height = search.sizeHint().height()
        size = self.size()
        search.setGeometry(0, def_height + 2, size.width(), src_height)
        margins.setTop(def_height + 2 + src_height)
        self.setViewportMargins(margins)

def format_desc(hint: VarHint) -> str:
    if False:
        print('Hello World!')
    'Describe the method and its parameters; used in list views and report'
    if hint is None:
        return Options[Methods.Default].short_desc
    desc = Options[hint.method_id].short_desc
    if hint.method_id == Methods.FixedWidthTime:
        (width, unit) = hint.args
        try:
            width = int(width)
        except ValueError:
            unit = f'{time_units[unit]}(s)'
        else:
            unit = f'{pl(width, time_units[unit])}'
        return desc.format(width, unit)
    return desc.format(*hint.args)

class DiscDomainModel(DomainModel):
    """
    Domain model that adds description of discretization methods and thresholds

    Also provides a tooltip that shows bins, that is, labels of the discretized
    variable.
    """

    def data(self, index, role=Qt.DisplayRole):
        if False:
            return 10
        if role == Qt.ToolTipRole:
            var = self[index.row()]
            data = index.data(Qt.UserRole)
            if not isinstance(data, DiscDesc):
                return super().data(index, role)
            tip = f'<b>{var.name}: </b>'
            values = map(html.escape, data.values)
            if not data.values:
                return None
            if len(data.values) <= 3:
                return f"""<p style="white-space:pre">{tip}{',&nbsp;&nbsp;'.join(values)}</p>"""
            else:
                return tip + '<br/>' + ''.join((f'- {value}<br/>' for value in values))
        value = super().data(index, role)
        if role == Qt.DisplayRole:
            try:
                (hint, points, values) = index.data(Qt.UserRole)
            except TypeError:
                pass
            else:
                value += f' ({format_desc(hint)}){points}'
        return value

class DefaultDiscModel(QAbstractListModel):
    """
    A model used for showing "Default settings" above the list view with var
    """
    icon = None

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        if DefaultDiscModel.icon is None:
            DefaultDiscModel.icon = gui.createAttributePixmap('★', QColor(0, 0, 0, 0), Qt.black)
        self.hint: VarHint = DefaultHint

    @staticmethod
    def rowCount(parent):
        if False:
            for i in range(10):
                print('nop')
        return 0 if parent.isValid() else 1

    @staticmethod
    def columnCount(parent):
        if False:
            return 10
        return 0 if parent.isValid() else 1

    def data(self, _, role=Qt.DisplayRole):
        if False:
            return 10
        if role == Qt.DisplayRole:
            return 'Default setting: ' + format_desc(self.hint)
        elif role == Qt.DecorationRole:
            return DefaultDiscModel.icon
        elif role == Qt.ToolTipRole:
            return 'Default setting for variables without specific setings'
        return None

    def setData(self, index, value, role=Qt.DisplayRole):
        if False:
            i = 10
            return i + 15
        if role == Qt.UserRole:
            self.hint = value
            self.dataChanged.emit(index, index)

class IncreasingNumbersListValidator(QValidator):
    """
    A validator for custom thresholds

    Requires a string with increasing comma-separated values. If the string
    ends with number followed by space, it inserts a comma.
    """

    @staticmethod
    def validate(string: str, pos: int) -> Tuple[QValidator.State, str, int]:
        if False:
            return 10
        for (i, c) in enumerate(string, start=1):
            if c not in '+-., 0123456789':
                return (QValidator.Invalid, string, i)
        prev = None
        if pos == len(string) >= 2 and string[-1] == ' ' and string[-2].isdigit():
            string = string[:-1] + ', '
            pos += 1
        for valuestr in re_custom_sep.split(string.strip()):
            try:
                value = float(valuestr)
            except ValueError:
                return (QValidator.Intermediate, string, pos)
            if prev is not None and value <= prev:
                return (QValidator.Intermediate, string, pos)
            prev = value
        return (QValidator.Acceptable, string, pos)

    @staticmethod
    def show_tip(widget: QWidget, pos: QPoint, text: str, timeout=-1, textFormat=Qt.AutoText, wordWrap=None):
        if False:
            return 10
        'Show a tooltip; used for invalid custom thresholds'
        propname = __name__ + '::show_tip_qlabel'
        if timeout < 0:
            timeout = widget.toolTipDuration()
        if timeout < 0:
            timeout = 5000 + 40 * max(0, len(text) - 100)
        tip = widget.property(propname)
        if not text and tip is None:
            return

        def hide():
            if False:
                while True:
                    i = 10
            w = tip.parent()
            w.setProperty(propname, None)
            tip.timer.stop()
            tip.close()
            tip.deleteLater()
        if not isinstance(tip, QLabel):
            tip = QLabel(objectName='tip-label', focusPolicy=Qt.NoFocus)
            tip.setBackgroundRole(QPalette.ToolTipBase)
            tip.setForegroundRole(QPalette.ToolTipText)
            tip.setPalette(QToolTip.palette())
            tip.setFont(QApplication.font('QTipLabel'))
            tip.setContentsMargins(2, 2, 2, 2)
            tip.timer = QTimer(tip, singleShot=True, objectName='hide-timer')
            tip.timer.timeout.connect(hide)
            widget.setProperty(propname, tip)
            tip.setParent(widget, Qt.ToolTip)
        tip.setText(text)
        tip.setTextFormat(textFormat)
        if wordWrap is None:
            wordWrap = textFormat != Qt.PlainText
        tip.setWordWrap(wordWrap)
        if not text:
            hide()
        else:
            tip.timer.start(timeout)
            tip.show()
            tip.move(pos)
from collections import namedtuple
globals().update(dict(DState=namedtuple('DState', ['method', 'points', 'disc_var'], defaults=(None, None)), Default=namedtuple('Default', ['method']), Leave=namedtuple('Leave', []), MDL=namedtuple('MDL', []), EqualFreq=namedtuple('EqualFreq', ['k']), EqualWidth=namedtuple('EqualWidth', ['k']), Remove=namedtuple('Remove', []), Custom=namedtuple('Custom', ['points'])))

class OWDiscretize(widget.OWWidget):
    name = 'Discretize'
    description = 'Discretize numeric variables'
    category = 'Transform'
    icon = 'icons/Discretize.svg'
    keywords = 'discretize, bin, categorical, nominal, ordinal'
    priority = 2130

    class Inputs:
        data = Input('Data', Table, doc='Input data table')

    class Outputs:
        data = Output('Data', Table, doc='Table with categorical features')
    settings_version = 3
    var_hints: Dict[KeyType, VarHint] = Setting({DefaultKey: DefaultHint}, schema_only=True)
    autosend = Setting(True)
    want_main_area = False

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.data = None
        self.discretized_vars: Dict[KeyType, DiscreteVariable] = {}
        self.__interface_update = False
        box = gui.hBox(self.controlArea, True, spacing=8)
        self._create_var_list(box)
        self._create_buttons(box)
        gui.auto_apply(self.buttonsArea, self, 'autosend')
        gui.rubber(self.buttonsArea)
        self.varview.select_default()

    def _create_var_list(self, box):
        if False:
            return 10
        'Create list view with variables'
        self.varview = ListViewSearch(selectionMode=QListView.ExtendedSelection, uniformItemSizes=True)
        self.varview.setModel(DiscDomainModel(valid_types=(ContinuousVariable, TimeVariable), order=DiscDomainModel.MIXED))
        self.varview.selectionModel().selectionChanged.connect(self._var_selection_changed)
        self.varview.default_view.selectionModel().selectionChanged.connect(self._default_selected)
        self._update_default_model()
        box.layout().addWidget(self.varview)

    def _create_buttons(self, box):
        if False:
            i = 10
            return i + 15
        'Create radio buttons'

        def intspin():
            if False:
                print('Hello World!')
            s = QSpinBox(self)
            s.setMinimum(2)
            s.setMaximum(10)
            s.setFixedWidth(60)
            s.setAlignment(Qt.AlignRight)
            s.setContentsMargins(0, 0, 0, 0)
            return (s, s.valueChanged)

        def widthline(validator):
            if False:
                for i in range(10):
                    print('nop')
            s = QLineEdit(self)
            s.setFixedWidth(60)
            s.setAlignment(Qt.AlignRight)
            s.setValidator(validator)
            s.setContentsMargins(0, 0, 0, 0)
            return (s, s.textChanged)

        def manual_cut_editline(text='', enabled=True) -> QLineEdit:
            if False:
                i = 10
                return i + 15
            edit = QLineEdit(text=text, placeholderText='e.g. 0.0, 0.5, 1.0', toolTip='<p style="white-space:pre">' + 'Enter cut points as a comma-separate list of \nstrictly increasing numbers e.g. 0.0, 0.5, 1.0).</p>', enabled=enabled)
            edit.setValidator(IncreasingNumbersListValidator())
            edit.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

            @edit.textChanged.connect
            def update():
                if False:
                    return 10
                validator = edit.validator()
                if validator is not None and edit.text().strip():
                    (state, _, _) = validator.validate(edit.text(), 0)
                else:
                    state = QValidator.Acceptable
                palette = edit.palette()
                colors = {QValidator.Intermediate: (Qt.yellow, Qt.black), QValidator.Invalid: (Qt.red, Qt.black)}.get(state, None)
                if colors is None:
                    palette = QPalette()
                else:
                    palette.setColor(QPalette.Base, colors[0])
                    palette.setColor(QPalette.Text, colors[1])
                cr = edit.cursorRect()
                p = edit.mapToGlobal(cr.bottomRight())
                edit.setPalette(palette)
                if state != QValidator.Acceptable and edit.isVisible():
                    validator.show_tip(edit, p, edit.toolTip(), textFormat=Qt.RichText)
                else:
                    validator.show_tip(edit, p, '')
            return (edit, edit.textChanged)
        children = []

        def button(id_, *controls, stretch=True):
            if False:
                for i in range(10):
                    print('nop')
            layout = QHBoxLayout()
            desc = Options[id_]
            button = QRadioButton(desc.label)
            button.setToolTip(desc.tooltip)
            self.button_group.addButton(button, id_)
            layout.addWidget(button)
            if controls:
                if stretch:
                    layout.addStretch(1)
                for (c, signal) in controls:
                    layout.addWidget(c)
                    if signal is not None:

                        @signal.connect
                        def arg_changed():
                            if False:
                                for i in range(10):
                                    print('nop')
                            self.button_group.button(id_).setChecked(True)
                            self.update_hints(id_)
            children.append(layout)
            button_box.layout().addLayout(layout)
            return (*controls, (None,))[0][0]
        button_box = gui.vBox(box)
        button_box.layout().setSpacing(0)
        button_box.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred))
        self.button_group = QButtonGroup(self)
        self.button_group.idClicked.connect(self.update_hints)
        button(Methods.Keep)
        button(Methods.Remove)
        self.binning_spin = button(Methods.Binning, intspin())
        validator = QDoubleValidator()
        validator.setBottom(0)
        self.width_line = button(Methods.FixedWidth, widthline(validator))
        self.width_time_unit = u = QComboBox(self)
        u.setContentsMargins(0, 0, 0, 0)
        u.addItems([f'{unit}(s)' for unit in time_units])
        validator = QIntValidator()
        validator.setBottom(1)
        self.width_time_line = button(Methods.FixedWidthTime, widthline(validator), (u, u.currentTextChanged))
        self.freq_spin = button(Methods.EqualFreq, intspin())
        self.width_spin = button(Methods.EqualWidth, intspin())
        button(Methods.MDL)
        self.copy_to_custom = FixedSizeButton(text='CC', toolTip='Copy the current cut points to manual mode')
        self.copy_to_custom.clicked.connect(self._copy_to_manual)
        self.threshold_line = button(Methods.Custom, manual_cut_editline(), (self.copy_to_custom, None), stretch=False)
        button(Methods.Default)
        maxheight = max((w.sizeHint().height() for w in children))
        for w in children:
            w.itemAt(0).widget().setFixedHeight(maxheight)
        button_box.layout().addStretch(1)

    def _update_default_model(self):
        if False:
            while True:
                i = 10
        'Update data in the model showing default settings'
        model = self.varview.default_view.model()
        model.setData(model.index(0), self.var_hints[DefaultKey], Qt.UserRole)

    def _set_mdl_button(self):
        if False:
            i = 10
            return i + 15
        'Disable MDL discretization for data with non-discrete class'
        mdl_button = self.button_group.button(Methods.MDL)
        if self.data is None or self.data.domain.has_discrete_class:
            mdl_button.setEnabled(True)
        else:
            if mdl_button.isChecked():
                self._check_button(Methods.Keep, True)
            mdl_button.setEnabled(False)

    def _check_button(self, method_id: Methods, checked: bool):
        if False:
            return 10
        'Checks the given button'
        self.button_group.button(method_id).setChecked(checked)

    def _uncheck_all_buttons(self):
        if False:
            for i in range(10):
                print('nop')
        'Uncheck all radio buttons'
        group = self.button_group
        button = group.checkedButton()
        if button is not None:
            group.setExclusive(False)
            button.setChecked(False)
            group.setExclusive(True)

    def _set_radio_enabled(self, method_id: Methods, value: bool):
        if False:
            for i in range(10):
                print('nop')
        'Enable/disable radio button and related controls'
        if self.button_group.button(method_id).isChecked() and (not value):
            self._uncheck_all_buttons()
        self.button_group.button(method_id).setEnabled(value)
        for control_name in Options[method_id].controls:
            getattr(self, control_name).setEnabled(value)

    def _get_values(self, method_id: Methods) -> Tuple[Union[int, float, str]]:
        if False:
            return 10
        'Return parameters from controls pertaining to the given method'
        controls = Options[method_id].controls
        values = []
        for control_name in controls:
            control = getattr(self, control_name)
            if isinstance(control, QSpinBox):
                values.append(control.value())
            elif isinstance(control, QComboBox):
                values.append(control.currentIndex())
            else:
                values.append(control.text())
        return tuple(values)

    def _set_values(self, method_id: Methods, values: Tuple[Union[str, int, float]]):
        if False:
            print('Hello World!')
        '\n        Set controls pertaining to the given method to parameters from hint\n        '
        controls = Options[method_id].controls
        for (control_name, value) in zip(controls, values):
            control = getattr(self, control_name)
            if isinstance(control, QSpinBox):
                control.setValue(value)
            elif isinstance(control, QComboBox):
                control.setCurrentIndex(value)
            else:
                control.setText(value)

    def varkeys_for_selection(self) -> List[KeyType]:
        if False:
            return 10
        "\n        Return list of KeyType's for selected variables (for indexing var_hints)\n\n        If 'Default settings' are selected, this returns DefaultKey\n        "
        model = self.varview.model()
        varkeys = [variable_key(model[index.row()]) for index in self.varview.selectionModel().selectedRows()]
        return varkeys or [DefaultKey]

    def update_hints(self, method_id: Methods):
        if False:
            for i in range(10):
                print('nop')
        '\n        Callback for radio buttons and for controls regulating parameters\n\n        This function:\n        - updates `var_hints` for all selected methods\n        - invalidates (removes) `discretized_vars` for affected variables\n        - calls _update_discretizations to compute and commit new discretization\n        - calls deferred commit\n\n        Data for list view models is updated in _update_discretizations\n        '
        if self.__interface_update:
            return
        method_id = Methods(method_id)
        args = self._get_values(method_id)
        keys = self.varkeys_for_selection()
        if method_id == Methods.Default:
            for key in keys:
                if key in self.var_hints:
                    del self.var_hints[key]
        else:
            self.var_hints.update(dict.fromkeys(keys, VarHint(method_id, args)))
        if keys == [DefaultKey]:
            invalidate = set(self.discretized_vars) - set(self.var_hints)
        else:
            invalidate = keys
        for key in invalidate:
            del self.discretized_vars[key]
        if keys == [DefaultKey]:
            self._update_default_model()
        self._update_discretizations()
        self.commit.deferred()

    def _update_discretizations(self):
        if False:
            i = 10
            return i + 15
        '\n        Compute invalidated (missing) discretizations\n\n        Also set data for list view models for all invalidated variables\n        '
        if self.data is None:
            return
        default_hint = self.var_hints[DefaultKey]
        model = self.varview.model()
        for (index, var) in enumerate(model):
            key = variable_key(var)
            if key in self.discretized_vars:
                continue
            var_hint = self.var_hints.get(key)
            (points, dvar) = self._discretize_var(var, var_hint or default_hint)
            self.discretized_vars[key] = dvar
            values = getattr(dvar, 'values', ())
            model.setData(model.index(index), DiscDesc(var_hint, points, values), Qt.UserRole)

    def _discretize_var(self, var: ContinuousVariable, hint: VarHint) -> Tuple[str, Optional[Variable]]:
        if False:
            return 10
        '\n        Discretize using method and data in the hint.\n\n        Returns a description (list of points or error/warning) and a\n        - discrete variable\n        - same variable (if kept numeric)\n        - None (if removed or errored)\n        '
        if isinstance(var, TimeVariable):
            if hint.method_id in (Methods.FixedWidth, Methods.Custom):
                return (': <keep, time var>', var)
        elif hint.method_id == Methods.FixedWidthTime:
            return (': <keep, not time>', var)
        function = Options[hint.method_id].function
        dvar = function(self.data, var, *hint.args)
        if isinstance(dvar, str):
            return (f' <{dvar}>', None)
        if dvar is None:
            return ('', None)
        elif dvar is var:
            return ('', var)
        thresholds = dvar.compute_value.points
        if len(thresholds) == 0:
            return (' <removed>', None)
        return (': ' + ', '.join(map(var.repr_val, thresholds)), dvar)

    def _copy_to_manual(self):
        if False:
            while True:
                i = 10
        '\n        Callback for \'CC\' button\n\n        Sets selected variables\' method to "Custom" and copies thresholds\n        to their VarHints. Variables that are not discretized (for any reason)\n        are skipped.\n\n        Discretizations are invalidated and then updated\n        (`_update_discretizations`).\n\n        If all selected variables have the same thresholds, it copies it to\n        the line edit. Otherwise it unchecks all radio buttons to keep the\n        interface consistent.\n        '
        varkeys = self.varkeys_for_selection()
        texts = set()
        for key in varkeys:
            dvar = self.discretized_vars.get(key)
            fmt = self.data.domain[key[0]].repr_val
            if isinstance(dvar, DiscreteVariable):
                text = ', '.join(map(fmt, dvar.compute_value.points))
                texts.add(text)
                self.var_hints[key] = VarHint(Methods.Custom, (text,))
                del self.discretized_vars[key]
        try:
            self.__interface_update = True
            if len(texts) == 1:
                self.threshold_line.setText(texts.pop())
            else:
                self._uncheck_all_buttons()
        finally:
            self.__interface_update = False
        self._update_discretizations()
        self.commit.deferred()

    def _default_selected(self, selected):
        if False:
            i = 10
            return i + 15
        "Callback for selecting 'Default setting'"
        if not selected:
            return
        self.varview.selectionModel().clearSelection()
        self._update_interface()
        set_enabled = self._set_radio_enabled
        set_enabled(Methods.Default, False)
        set_enabled(Methods.FixedWidth, True)
        set_enabled(Methods.FixedWidthTime, True)
        set_enabled(Methods.Custom, True)
        self.copy_to_custom.setEnabled(False)

    def _var_selection_changed(self, _):
        if False:
            return 10
        'Callback for changed selection in listview with variables'
        selected = self.varview.selectionModel().selectedIndexes()
        if not selected:
            return
        self.varview.default_view.selectionModel().clearSelection()
        self._update_interface()
        set_enabled = self._set_radio_enabled
        vars_ = [self.data.domain[name] for (name, _) in self.varkeys_for_selection()]
        no_time = not any((isinstance(var, TimeVariable) for var in vars_))
        all_time = all((isinstance(var, TimeVariable) for var in vars_))
        set_enabled(Methods.Default, True)
        set_enabled(Methods.FixedWidth, no_time)
        set_enabled(Methods.Custom, no_time)
        self.copy_to_custom.setEnabled(no_time)
        set_enabled(Methods.FixedWidthTime, all_time)

    def _update_interface(self):
        if False:
            print('Hello World!')
        '\n        Update the user interface according to selection\n\n        - If VarHints for all selected variables are the same, check the\n          corresponding radio button and fill the corresponding controls;\n        - otherwise, uncheck all radios.\n        '
        if self.__interface_update:
            return
        try:
            self.__interface_update = True
            keys = self.varkeys_for_selection()
            mset = list(unique_everseen(map(self.var_hints.get, keys)))
            if len(mset) != 1:
                self._uncheck_all_buttons()
                return
            if mset == [None]:
                (method_id, args) = (Methods.Default, ())
            else:
                (method_id, args) = mset.pop()
            self._check_button(method_id, True)
            self._set_values(method_id, args)
        finally:
            self.__interface_update = False

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        if False:
            for i in range(10):
                print('nop')
        self.discretized_vars = {}
        self.data = data
        self.varview.model().set_domain(None if data is None else data.domain)
        self._update_discretizations()
        self._update_default_model()
        self.varview.select_default()
        self._set_mdl_button()
        self.commit.now()

    @gui.deferred
    def commit(self):
        if False:
            i = 10
            return i + 15
        if self.data is None:
            self.Outputs.data.send(None)
            return

        def part(variables: List[Variable]) -> List[Variable]:
            if False:
                while True:
                    i = 10
            return [dvar for dvar in (self.discretized_vars.get(variable_key(v), v) for v in variables) if dvar]
        d = self.data.domain
        domain = Domain(part(d.attributes), part(d.class_vars), part(d.metas))
        output = self.data.transform(domain)
        self.Outputs.data.send(output)

    def send_report(self):
        if False:
            i = 10
            return i + 15
        dmodel = self.varview.default_view.model()
        desc = dmodel.data(dmodel.index(0))
        self.report_items((tuple(desc.split(': ', maxsplit=1)),))
        model = self.varview.model()
        reported = []
        for row in range(model.rowCount()):
            name = model[row].name
            desc = model.data(model.index(row), Qt.UserRole)
            if desc.hint is not None:
                name = f'{name} ({format_desc(desc.hint)})'
            reported.append((name, ', '.join(desc.values)))
        self.report_items('Variables', reported)

    @classmethod
    def migrate_settings(cls, settings, version):
        if False:
            while True:
                i = 10
        if version is None or version < 2:
            default = settings.pop('default_method', 0)
            default = Methods(default + 1)
            settings['default_method_name'] = default.name
        if version is None or version < 3:
            method_name = settings.pop('default_method_name', DefaultHint.method_id.name)
            k = settings.pop('default_k', 3)
            cut_points = settings.pop('default_cutpoints', ())
            method_id = getattr(Methods, method_name)
            if method_id in (Methods.EqualFreq, Methods.EqualWidth):
                args = (k,)
            elif method_id == Methods.Custom:
                args = (cut_points,)
            else:
                args = ()
            default_hint = VarHint(method_id, args)
            var_hints = {DefaultKey: default_hint}
            for context in settings.pop('context_settings', []):
                values = context.values
                if 'saved_var_states' not in values:
                    continue
                (var_states, _) = values.pop('saved_var_states')
                for ((tpe, name), dstate) in var_states.items():
                    key = (name, tpe == 4)
                    method = dstate.method
                    method_name = type(method).__name__.replace('Leave', 'Keep')
                    if method_name == 'Default':
                        continue
                    if method_name == 'Custom':
                        args = (', '.join((f'{x:g}' for x in method.points)),)
                    else:
                        args = tuple(method)
                    var_hints[key] = VarHint(getattr(Methods, method_name), args)
            settings['var_hints'] = var_hints
if __name__ == '__main__':
    WidgetPreview(OWDiscretize).run(Table('heart_disease'))