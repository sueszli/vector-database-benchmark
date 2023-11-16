import os
from itertools import chain
import json
import numpy as np
from AnyQt.QtCore import Qt, QSize, QAbstractTableModel, QModelIndex, QTimer, QSettings
from AnyQt.QtGui import QColor, QFont, QBrush
from AnyQt.QtWidgets import QHeaderView, QColorDialog, QTableView, QComboBox, QFileDialog, QMessageBox
from orangewidget.settings import IncompatibleContext
import Orange
from Orange.preprocess.transformation import Identity
from Orange.util import color_to_hex, hex_to_color
from Orange.widgets import widget, settings, gui
from Orange.widgets.gui import HorizontalGridDelegate
from Orange.widgets.utils import itemmodels, colorpalettes
from Orange.widgets.utils.localization import pl
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.report import colored_square as square
from Orange.widgets.widget import Input, Output
ColorRole = next(gui.OrangeUserRole)
StripRole = next(gui.OrangeUserRole)

class InvalidFileFormat(Exception):
    pass

class AttrDesc:
    """
    Describes modifications that will be applied to variable.

    Provides methods that return either the modified value or the original

    Attributes:
        var (Variable): an instance of variable
        new_name (str or `None`): a changed name or `None`
    """

    def __init__(self, var):
        if False:
            for i in range(10):
                print('nop')
        self.var = var.copy(compute_value=None)
        self.new_name = None

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.new_name = None

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return self.new_name or self.var.name

    @name.setter
    def name(self, name):
        if False:
            i = 10
            return i + 15
        self.new_name = name

    def to_dict(self):
        if False:
            print('Hello World!')
        d = {}
        if self.new_name is not None:
            d['rename'] = self.new_name
        return d

    @classmethod
    def from_dict(cls, var, data):
        if False:
            print('Hello World!')
        desc = cls(var)
        if not isinstance(data, dict):
            raise InvalidFileFormat
        new_name = data.get('rename')
        if new_name is not None:
            if not isinstance(new_name, str):
                raise InvalidFileFormat
            desc.name = new_name
        return (desc, [])

class DiscAttrDesc(AttrDesc):
    """
    Describes modifications that will be applied to variable.

    Provides methods that return either the modified value or the original

    Attributes:
        var (DiscreteVariable): an instance of variable
        name (str or `None`): a changed name or `None`
        new_colors (list of tuple or None): new colors as tuples (R, G, B)
        new_values (list of str or None): new names for values, if changed
    """

    def __init__(self, var):
        if False:
            i = 10
            return i + 15
        super().__init__(var)
        self.new_colors = None
        self.new_values = None

    def reset(self):
        if False:
            print('Hello World!')
        super().reset()
        self.new_colors = None
        self.new_values = None

    @property
    def colors(self):
        if False:
            i = 10
            return i + 15
        if self.new_colors is None:
            return self.var.colors
        else:
            return self.new_colors

    def set_color(self, i, color):
        if False:
            while True:
                i = 10
        if self.new_colors is None:
            self.new_colors = list(self.var.colors)
        self.new_colors[i] = color

    @property
    def values(self):
        if False:
            i = 10
            return i + 15
        return tuple(self.new_values or self.var.values)

    def set_value(self, i, value):
        if False:
            for i in range(10):
                print('nop')
        if not self.new_values:
            self.new_values = list(self.var.values)
        self.new_values[i] = value

    def create_variable(self, base_var):
        if False:
            while True:
                i = 10
        new_var = base_var.copy(name=self.name, values=self.values, compute_value=Identity(base_var))
        new_var.colors = np.asarray(self.colors)
        return new_var

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        d = super().to_dict()
        if self.new_values is not None:
            d['renamed_values'] = {k: v for (k, v) in zip(self.var.values, self.new_values) if k != v}
        if self.new_colors is not None:
            d['colors'] = {value: color_to_hex(color) for (value, color) in zip(self.var.values, self.colors)}
        return d

    @classmethod
    def from_dict(cls, var, data):
        if False:
            while True:
                i = 10

        def _check_dict_str_str(d):
            if False:
                print('Hello World!')
            if not isinstance(d, dict) or not all((isinstance(val, str) for val in chain(d, d.values()))):
                raise InvalidFileFormat
        (obj, warnings) = super().from_dict(var, data)
        val_map = data.get('renamed_values')
        if val_map is not None:
            _check_dict_str_str(val_map)
            mapped_values = [val_map.get(value, value) for value in var.values]
            if len(set(mapped_values)) != len(mapped_values):
                warnings.append(f'{var.name}: renaming of values ignored due to duplicate names')
            else:
                obj.new_values = mapped_values
        new_colors = data.get('colors')
        if new_colors is not None:
            _check_dict_str_str(new_colors)
            colors = []
            for (value, def_color) in zip(var.values, var.palette.palette):
                if value in new_colors:
                    try:
                        color = hex_to_color(new_colors[value])
                    except ValueError as exc:
                        raise InvalidFileFormat from exc
                    colors.append(color)
                else:
                    colors.append(def_color)
                obj.new_colors = colors
        return (obj, warnings)

class ContAttrDesc(AttrDesc):
    """
    Describes modifications that will be applied to variable.

    Provides methods that return either the modified value or the original

    Attributes:
        var (ContinuousVariable): an instance of variable
        name (str or `None`): a changed name or `None`
        palette_name (str or None): name of palette or None if unmodified
    """

    def __init__(self, var):
        if False:
            while True:
                i = 10
        super().__init__(var)
        self.new_palette_name = self._default_palette_name()

    def reset(self):
        if False:
            print('Hello World!')
        super().reset()
        self.new_palette_name = self._default_palette_name()

    def _default_palette_name(self):
        if False:
            print('Hello World!')
        if self.var.palette.name not in colorpalettes.ContinuousPalettes:
            return colorpalettes.DefaultContinuousPaletteName
        else:
            return None

    @property
    def palette_name(self):
        if False:
            return 10
        return self.new_palette_name or self.var.palette.name

    @palette_name.setter
    def palette_name(self, palette_name):
        if False:
            for i in range(10):
                print('nop')
        self.new_palette_name = palette_name

    def create_variable(self, base_var):
        if False:
            print('Hello World!')
        new_var = base_var.copy(name=self.name, compute_value=Identity(base_var))
        new_var.attributes['palette'] = self.palette_name
        return new_var

    def to_dict(self):
        if False:
            print('Hello World!')
        d = super().to_dict()
        if self.new_palette_name is not None:
            d['colors'] = self.palette_name
        return d

    @classmethod
    def from_dict(cls, var, data):
        if False:
            i = 10
            return i + 15
        (obj, warnings) = super().from_dict(var, data)
        colors = data.get('colors')
        if colors is not None:
            if colors not in colorpalettes.ContinuousPalettes:
                raise InvalidFileFormat
            obj.palette_name = colors
        return (obj, warnings)

class ColorTableModel(QAbstractTableModel):
    """
    Base color model for discrete and continuous variables. The model handles:
    - the first column - variable name (including setData)
    - flags
    - row count, computed as len(attrdescs)

    Attribute:
        attrdescs (list of AttrDesc): attrdescs with user-defined changes
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        QAbstractTableModel.__init__(self)
        self.attrdescs = []

    @staticmethod
    def flags(_):
        if False:
            i = 10
            return i + 15
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def set_data(self, attrdescs):
        if False:
            while True:
                i = 10
        self.modelAboutToBeReset.emit()
        self.attrdescs = attrdescs
        self.modelReset.emit()

    def rowCount(self, parent=QModelIndex()):
        if False:
            while True:
                i = 10
        return 0 if parent.isValid() else len(self.attrdescs)

    def data(self, index, role=Qt.DisplayRole):
        if False:
            for i in range(10):
                print('nop')
        row = index.row()
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self.attrdescs[row].name
        if role == Qt.FontRole:
            font = QFont()
            font.setBold(True)
            return font
        if role == Qt.TextAlignmentRole:
            return Qt.AlignRight | Qt.AlignVCenter
        return None

    def setData(self, index, value, role):
        if False:
            for i in range(10):
                print('nop')
        if role == Qt.EditRole:
            self.attrdescs[index.row()].name = value
        else:
            return False
        self.dataChanged.emit(index, index)
        return True

    def reset(self):
        if False:
            return 10
        self.beginResetModel()
        for desc in self.attrdescs:
            desc.reset()
        self.endResetModel()

class DiscColorTableModel(ColorTableModel):
    """
    A model that stores the colors corresponding to values of discrete
    variables. Colors are shown as decorations.
    """

    def columnCount(self, parent=QModelIndex()):
        if False:
            for i in range(10):
                print('nop')
        if parent.isValid():
            return 0
        return 1 + max((len(row.var.values) for row in self.attrdescs), default=0)

    def data(self, index, role=Qt.DisplayRole):
        if False:
            for i in range(10):
                print('nop')
        (row, col) = (index.row(), index.column())
        if col == 0:
            return super().data(index, role)
        desc = self.attrdescs[row]
        if col > len(desc.var.values):
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return desc.values[col - 1]
        color = desc.colors[col - 1]
        if role == Qt.DecorationRole:
            return QColor(*color)
        if role == Qt.ToolTipRole:
            return color_to_hex(color)
        if role == ColorRole:
            return color
        return None

    def setData(self, index, value, role):
        if False:
            print('Hello World!')
        (row, col) = (index.row(), index.column())
        if col == 0:
            return super().setData(index, value, role)
        desc = self.attrdescs[row]
        if role == ColorRole:
            desc.set_color(col - 1, value[:3])
        elif role == Qt.EditRole:
            desc.set_value(col - 1, value)
        else:
            return False
        self.dataChanged.emit(index, index)
        return True

class ContColorTableModel(ColorTableModel):
    """A model that stores the colors corresponding to values of discrete
    variables. Colors are shown as decorations.

    Attributes:
        mouse_row (int): the row over which the mouse is hovering
    """

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.mouse_row = None

    def set_mouse_row(self, row):
        if False:
            while True:
                i = 10
        self.mouse_row = row

    @staticmethod
    def columnCount(parent=QModelIndex()):
        if False:
            i = 10
            return i + 15
        return 0 if parent.isValid() else 3

    def data(self, index, role=Qt.DisplayRole):
        if False:
            return 10

        def _column0():
            if False:
                i = 10
                return i + 15
            return ColorTableModel.data(self, index, role)

        def _column1():
            if False:
                while True:
                    i = 10
            palette = colorpalettes.ContinuousPalettes[desc.palette_name]
            if role == Qt.ToolTipRole:
                return palette.friendly_name
            if role == ColorRole:
                return palette
            if role == StripRole:
                return palette.color_strip(128, 16)
            if role == Qt.SizeHintRole:
                return QSize(150, 16)
            return None

        def _column2():
            if False:
                i = 10
                return i + 15
            if role == Qt.SizeHintRole:
                return QSize(100, 1)
            if role == Qt.ForegroundRole:
                return QBrush(Qt.blue)
            if row == self.mouse_row and role == Qt.DisplayRole:
                return 'Copy to all'
            return None
        (row, col) = (index.row(), index.column())
        desc = self.attrdescs[row]
        if 0 <= col <= 2:
            return [_column0, _column1, _column2][col]()

    def setData(self, index, value, role):
        if False:
            print('Hello World!')
        (row, col) = (index.row(), index.column())
        if col == 0:
            return super().setData(index, value, role)
        if role == ColorRole:
            self.attrdescs[row].palette_name = value.name
        else:
            return False
        self.dataChanged.emit(index, index)
        return True

    def copy_to_all(self, index):
        if False:
            i = 10
            return i + 15
        palette_name = self.attrdescs[index.row()].palette_name
        for desc in self.attrdescs:
            desc.palette_name = palette_name
        self.dataChanged.emit(self.index(0, 1), self.index(self.rowCount(), 1))

class ColorStripDelegate(HorizontalGridDelegate):

    def __init__(self, view):
        if False:
            print('Hello World!')
        super().__init__()
        self.view = view

    def createEditor(self, parent, _, index):
        if False:
            for i in range(10):
                print('nop')

        class Combo(QComboBox):

            def __init__(self, parent, initial_data, view):
                if False:
                    print('Hello World!')
                super().__init__(parent)
                model = itemmodels.ContinuousPalettesModel(icon_width=128)
                self.setModel(model)
                self.setCurrentIndex(model.indexOf(initial_data))
                self.setIconSize(QSize(128, 16))
                QTimer.singleShot(0, self.showPopup)
                self.view = view

            def hidePopup(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().hidePopup()
                self.view.closeEditor(self, ColorStripDelegate.NoHint)

        def select(i):
            if False:
                print('Hello World!')
            self.view.model().setData(index, combo.model().index(i, 0).data(Qt.UserRole), ColorRole)
        combo = Combo(parent, index.data(ColorRole), self.view)
        combo.currentIndexChanged[int].connect(select)
        return combo

    def paint(self, painter, option, index):
        if False:
            print('Hello World!')
        strip = index.data(StripRole)
        rect = option.rect
        painter.drawPixmap(rect.x() + 13, int(rect.y() + (rect.height() - strip.height()) / 2), strip)
        super().paint(painter, option, index)

class ColorTable(QTableView):
    """
    The base table view for discrete and continuous attributes.

    Sets the basic properties of the table and implementes mouseRelease that
    calls handle_click with appropriate index. It also prepares a grid_deleagte
    that is used in derived classes.
    """

    def __init__(self, model):
        if False:
            print('Hello World!')
        QTableView.__init__(self)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setSelectionMode(QTableView.NoSelection)
        self.setModel(model)
        self.grid_delegate = HorizontalGridDelegate()

    def mouseReleaseEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        index = self.indexAt(event.pos())
        if not index.isValid():
            return
        rect = self.visualRect(index)
        self.handle_click(index, event.pos().x() - rect.x())

class DiscreteTable(ColorTable):
    """Table view for discrete variables"""

    def __init__(self, model):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(model)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setItemDelegate(self.grid_delegate)
        self.setEditTriggers(QTableView.NoEditTriggers)

    def handle_click(self, index, x_offset):
        if False:
            i = 10
            return i + 15
        '\n        Handle click events for the first column (call the edit method)\n        and the second (call method for changing the palette)\n        '
        if self.model().data(index, Qt.EditRole) is None:
            return
        if index.column() == 0 or x_offset > 24:
            self.edit(index)
        else:
            self.change_color(index)

    def change_color(self, index):
        if False:
            while True:
                i = 10
        'Invoke palette editor and set the color'
        color = self.model().data(index, ColorRole)
        if color is None:
            return
        dlg = QColorDialog(QColor(*color))
        if dlg.exec():
            color = dlg.selectedColor()
            self.model().setData(index, color.getRgb(), ColorRole)

class ContinuousTable(ColorTable):
    """Table view for continuous variables"""

    def __init__(self, model):
        if False:
            return 10
        super().__init__(model)
        self.viewport().setMouseTracking(True)
        self.color_delegate = ColorStripDelegate(self)
        self.setItemDelegateForColumn(0, self.grid_delegate)
        self.setItemDelegateForColumn(1, self.color_delegate)
        self.setColumnWidth(1, 256)
        self.setEditTriggers(QTableView.SelectedClicked | QTableView.DoubleClicked)

    def mouseMoveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Store the hovered row index in the model, trigger viewport update'
        pos = event.pos()
        ind = self.indexAt(pos)
        self.model().set_mouse_row(ind.row())
        super().mouseMoveEvent(event)
        self.viewport().update()

    def leaveEvent(self, _):
        if False:
            for i in range(10):
                print('nop')
        'Remove the stored the hovered row index, trigger viewport update'
        self.model().set_mouse_row(None)
        self.viewport().update()

    def handle_click(self, index, _):
        if False:
            i = 10
            return i + 15
        'Call the specific methods for handling clicks for each column'
        if index.column() < 2:
            self.edit(index)
        elif index.column() == 2:
            self.model().copy_to_all(index)

class OWColor(widget.OWWidget):
    name = 'Color'
    description = 'Set color legend for variables.'
    icon = 'icons/Colors.svg'

    class Inputs:
        data = Input('Data', Orange.data.Table)

    class Outputs:
        data = Output('Data', Orange.data.Table)
    settingsHandler = settings.PerfectDomainContextHandler(match_values=settings.PerfectDomainContextHandler.MATCH_VALUES_ALL)
    disc_descs = settings.ContextSetting([])
    cont_descs = settings.ContextSetting([])
    selected_schema_index = settings.Setting(0)
    auto_apply = settings.Setting(True)
    settings_version = 2
    want_main_area = False

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.data = None
        self.orig_domain = self.domain = None
        box = gui.hBox(self.controlArea, 'Discrete Variables')
        self.disc_model = DiscColorTableModel()
        self.disc_view = DiscreteTable(self.disc_model)
        self.disc_model.dataChanged.connect(self._on_data_changed)
        box.layout().addWidget(self.disc_view)
        box = gui.hBox(self.controlArea, 'Numeric Variables')
        self.cont_model = ContColorTableModel()
        self.cont_view = ContinuousTable(self.cont_model)
        self.cont_model.dataChanged.connect(self._on_data_changed)
        box.layout().addWidget(self.cont_view)
        box = gui.hBox(self.buttonsArea)
        gui.button(box, self, 'Save', callback=self.save)
        gui.button(box, self, 'Load', callback=self.load)
        gui.button(box, self, 'Reset', callback=self.reset)
        gui.rubber(self.buttonsArea)
        gui.auto_apply(self.buttonsArea, self, 'auto_apply')

    @staticmethod
    def sizeHint():
        if False:
            while True:
                i = 10
        return QSize(500, 570)

    @Inputs.data
    def set_data(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.closeContext()
        self.disc_descs = []
        self.cont_descs = []
        if data is None:
            self.data = self.domain = None
        else:
            self.data = data
            for var in chain(data.domain.variables, data.domain.metas):
                if var.is_discrete:
                    self.disc_descs.append(DiscAttrDesc(var))
                elif var.is_continuous:
                    self.cont_descs.append(ContAttrDesc(var))
        self.disc_model.set_data(self.disc_descs)
        self.cont_model.set_data(self.cont_descs)
        self.openContext(data)
        self.disc_view.resizeColumnsToContents()
        self.cont_view.resizeColumnsToContents()
        self.commit.now()

    def _on_data_changed(self):
        if False:
            print('Hello World!')
        self.commit.deferred()

    def reset(self):
        if False:
            while True:
                i = 10
        self.disc_model.reset()
        self.cont_model.reset()
        self.commit.now()

    def save(self):
        if False:
            while True:
                i = 10
        (fname, _) = QFileDialog.getSaveFileName(self, 'File name', self._start_dir(), 'Variable definitions (*.colors)')
        if not fname:
            return
        QSettings().setValue('colorwidget/last-location', os.path.split(fname)[0])
        self._save_var_defs(fname)

    def _save_var_defs(self, fname):
        if False:
            i = 10
            return i + 15
        with open(fname, 'w') as f:
            json.dump({vartype: {var.name: var_data for (var, var_data) in ((desc.var, desc.to_dict()) for desc in repo) if var_data} for (vartype, repo) in (('categorical', self.disc_descs), ('numeric', self.cont_descs))}, f, indent=4)

    def load(self):
        if False:
            while True:
                i = 10
        (fname, _) = QFileDialog.getOpenFileName(self, 'File name', self._start_dir(), 'Variable definitions (*.colors)')
        if not fname:
            return
        try:
            with open(fname) as f:
                js = json.load(f)
                self._parse_var_defs(js)
        except IOError:
            QMessageBox.critical(self, 'File error', 'File cannot be opened.')
            return
        except (json.JSONDecodeError, InvalidFileFormat):
            QMessageBox.critical(self, 'File error', 'Invalid file format.')
            return

    def _parse_var_defs(self, js):
        if False:
            i = 10
            return i + 15
        if not isinstance(js, dict) or set(js) != {'categorical', 'numeric'}:
            raise InvalidFileFormat
        try:
            renames = {var_name: desc['rename'] for repo in js.values() for (var_name, desc) in repo.items() if 'rename' in desc}
        except Exception as exc:
            raise InvalidFileFormat from exc
        if not all((isinstance(val, str) for val in chain(renames, renames.values()))):
            raise InvalidFileFormat
        renamed_vars = {renames.get(desc.var.name, desc.var.name) for desc in chain(self.disc_descs, self.cont_descs)}
        if len(renamed_vars) != len(self.disc_descs) + len(self.cont_descs):
            QMessageBox.warning(self, 'Duplicated variable names', 'Variables will not be renamed due to duplicated names.')
            for repo in js.values():
                for desc in repo.values():
                    desc.pop('rename', None)
        unused_vars = []
        both_descs = []
        warnings = []
        for (old_desc, repo, desc_type) in ((self.disc_descs, 'categorical', DiscAttrDesc), (self.cont_descs, 'numeric', ContAttrDesc)):
            var_by_name = {desc.var.name: desc.var for desc in old_desc}
            new_descs = {}
            for (var_name, var_data) in js[repo].items():
                var = var_by_name.get(var_name)
                if var is None:
                    unused_vars.append(var_name)
                    continue
                (new_descs[var_name], warn) = desc_type.from_dict(var, var_data)
                warnings += warn
            both_descs.append(new_descs)
        if unused_vars:
            names = [f"'{name}'" for name in unused_vars]
            if len(unused_vars) == 1:
                warn = f'Definition for variable {names[0]}, which does not appear in the data, was ignored.\n'
            else:
                if len(unused_vars) <= 5:
                    warn = f"Definitions for variables {', '.join(names[:-1])} and {names[-1]}"
                else:
                    warn = f"Definitions for {', '.join(names[:4])} and {len(names) - 4} other variables"
                warn += ', which do not appear in the data, were ignored.\n'
            warnings.insert(0, warn)
        self.disc_descs = [both_descs[0].get(desc.var.name, desc) for desc in self.disc_descs]
        self.cont_descs = [both_descs[1].get(desc.var.name, desc) for desc in self.cont_descs]
        if warnings:
            QMessageBox.warning(self, 'Invalid definitions', '\n'.join(warnings))
        self.disc_model.set_data(self.disc_descs)
        self.cont_model.set_data(self.cont_descs)
        self.commit.now()

    def _start_dir(self):
        if False:
            print('Hello World!')
        return self.workflowEnv().get('basedir') or QSettings().value('colorwidget/last-location') or os.path.expanduser(f'~{os.sep}')

    @gui.deferred
    def commit(self):
        if False:
            for i in range(10):
                print('nop')

        def make(variables):
            if False:
                return 10
            new_vars = []
            for var in variables:
                source = disc_dict if var.is_discrete else cont_dict
                desc = source.get(var.name)
                new_vars.append(desc.create_variable(var) if desc else var)
            return new_vars
        if self.data is None:
            self.Outputs.data.send(None)
            return
        disc_dict = {desc.var.name: desc for desc in self.disc_descs}
        cont_dict = {desc.var.name: desc for desc in self.cont_descs}
        dom = self.data.domain
        new_domain = Orange.data.Domain(make(dom.attributes), make(dom.class_vars), make(dom.metas))
        new_data = self.data.transform(new_domain)
        self.Outputs.data.send(new_data)

    def send_report(self):
        if False:
            for i in range(10):
                print('nop')
        'Send report'

        def _report_variables(variables):
            if False:
                while True:
                    i = 10

            def was(n, o):
                if False:
                    print('Hello World!')
                return n if n == o else f'{n} (was: {o})'
            max_values = max((len(var.values) for var in variables if var.is_discrete), default=1)
            rows = ''
            disc_dict = {k.var.name: k for k in self.disc_descs}
            cont_dict = {k.var.name: k for k in self.cont_descs}
            for var in variables:
                if var.is_discrete:
                    desc = disc_dict[var.name]
                    value_cols = '    \n'.join((f'<td>{square(*color)} {was(value, old_value)}</td>' for (color, value, old_value) in zip(desc.colors, desc.values, var.values)))
                elif var.is_continuous:
                    desc = cont_dict[var.name]
                    pal = colorpalettes.ContinuousPalettes[desc.palette_name]
                    value_cols = f'<td colspan="{max_values}">{pal.friendly_name}</td>'
                else:
                    continue
                names = was(desc.name, desc.var.name)
                rows += f'<tr style="height: 2em">\n  <th style="text-align: right">{names}</th>  {value_cols}\n</tr>\n'
            return rows
        if not self.data:
            return
        dom = self.data.domain
        sections = ((name, _report_variables(variables)) for (name, variables) in (('Features', dom.attributes), (f"{pl(len(dom.class_vars), 'Outcome')}", dom.class_vars), ('Meta attributes', dom.metas)))
        table = ''.join((f'<tr><th>{name}</th></tr>{rows}' for (name, rows) in sections if rows))
        if table:
            self.report_raw(f'<table>{table}</table>')

    @classmethod
    def migrate_context(cls, _, version):
        if False:
            i = 10
            return i + 15
        if not version or version < 2:
            raise IncompatibleContext
if __name__ == '__main__':
    WidgetPreview(OWColor).run(Orange.data.Table('heart_disease.tab'))