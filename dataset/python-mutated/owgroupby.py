from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Set
import pandas as pd
from numpy import nan
from AnyQt.QtCore import QAbstractTableModel, QEvent, QItemSelectionModel, QModelIndex, Qt
from AnyQt.QtWidgets import QAbstractItemView, QCheckBox, QGridLayout, QHeaderView, QTableView
from orangewidget.settings import ContextSetting, Setting
from orangewidget.utils.listview import ListViewFilter
from orangewidget.utils.signals import Input, Output
from orangewidget.utils import enum_as_int
from orangewidget.widget import Msg
from pandas.core.dtypes.common import is_datetime64_any_dtype
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table, TimeVariable, Variable
from Orange.data.aggregate import OrangeTableGroupBy
from Orange.util import wrap_callback
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget
Aggregation = namedtuple('Aggregation', ['function', 'types'])

def concatenate(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Concatenate values of series if value is not missing (nan or empty string\n    for StringVariable)\n    '
    return ' '.join((str(v) for v in x if not pd.isnull(v) and len(str(v)) > 0))

def std(s):
    if False:
        print('Hello World!')
    "\n    Std that also handle time variable. Pandas's std return Timedelta object in\n    case of datetime columns - transform TimeDelta to seconds\n    "
    std_ = s.std()
    if isinstance(std_, pd.Timedelta):
        return std_.total_seconds()
    return nan if pd.isna(std_) else std_

def var(s):
    if False:
        return 10
    "\n    Variance that also handle time variable. Pandas's variance function somehow\n    doesn't support DateTimeArray - this function fist converts datetime series\n    to UNIX epoch and then computes variance\n    "
    if is_datetime64_any_dtype(s):
        initial_ts = pd.Timestamp('1970-01-01', tz=None if s.dt.tz is None else 'UTC')
        s = (s - initial_ts) / pd.Timedelta('1s')
    var_ = s.var()
    return var_.total_seconds() if isinstance(var_, pd.Timedelta) else var_

def span(s):
    if False:
        while True:
            i = 10
    '\n    Span that also handle time variable. Time substitution return Timedelta\n    object in case of datetime columns - transform TimeDelta to seconds\n    '
    span_ = pd.Series.max(s) - pd.Series.min(s)
    return span_.total_seconds() if isinstance(span_, pd.Timedelta) else span_
AGGREGATIONS = {'Mean': Aggregation('mean', {ContinuousVariable, TimeVariable}), 'Median': Aggregation('median', {ContinuousVariable, TimeVariable}), 'Q1': Aggregation(lambda s: s.quantile(0.25), {ContinuousVariable, TimeVariable}), 'Q3': Aggregation(lambda s: s.quantile(0.75), {ContinuousVariable, TimeVariable}), 'Min. value': Aggregation('min', {ContinuousVariable, TimeVariable}), 'Max. value': Aggregation('max', {ContinuousVariable, TimeVariable}), 'Mode': Aggregation(lambda x: pd.Series.mode(x).get(0, nan), {ContinuousVariable, DiscreteVariable, TimeVariable}), 'Standard deviation': Aggregation(std, {ContinuousVariable, TimeVariable}), 'Variance': Aggregation(var, {ContinuousVariable, TimeVariable}), 'Sum': Aggregation('sum', {ContinuousVariable}), 'Concatenate': Aggregation(concatenate, {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable}), 'Span': Aggregation(span, {ContinuousVariable, TimeVariable}), 'First value': Aggregation('first', {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable}), 'Last value': Aggregation('last', {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable}), 'Random value': Aggregation(lambda x: x.sample(1, random_state=0), {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable}), 'Count defined': Aggregation('count', {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable}), 'Count': Aggregation('size', {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable}), 'Proportion defined': Aggregation(lambda x: x.count() / x.size, {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable})}
AGGREGATIONS_ORD = list(AGGREGATIONS)
DEFAULT_AGGREGATIONS = {var: {next((name for (name, agg) in AGGREGATIONS.items() if var in agg.types))} for var in (ContinuousVariable, TimeVariable, DiscreteVariable, StringVariable)}

@dataclass
class Result:
    group_by: OrangeTableGroupBy = None
    result_table: Optional[Table] = None

def _run(data: Table, group_by_attrs: List[Variable], aggregations: Dict[Variable, Set[str]], result: Result, state: TaskState) -> Result:
    if False:
        return 10

    def progress(part):
        if False:
            print('Hello World!')
        state.set_progress_value(part * 100)
        if state.is_interruption_requested():
            raise Exception
    state.set_status('Aggregating')
    if result.group_by is None:
        result.group_by = data.groupby(group_by_attrs)
    state.set_partial_result(result)
    aggregations = {var: [(agg, AGGREGATIONS[agg].function) for agg in sorted(aggs, key=AGGREGATIONS_ORD.index)] for (var, aggs) in aggregations.items()}
    result.result_table = result.group_by.aggregate(aggregations, wrap_callback(progress, 0.2, 1))
    return result

class TabColumn:
    attribute = 0
    aggregations = 1
TABLE_COLUMN_NAMES = ['Attributes', 'Aggregations']

class VarTableModel(QAbstractTableModel):

    def __init__(self, parent: 'OWGroupBy', *args):
        if False:
            return 10
        super().__init__(*args)
        self.domain = None
        self.parent = parent

    def set_domain(self, domain: Domain) -> None:
        if False:
            return 10
        '\n        Reset the table view to new domain\n        '
        self.domain = domain
        self.modelReset.emit()

    def update_aggregation(self, attribute: str) -> None:
        if False:
            print('Hello World!')
        '\n        Reset the aggregation values in the table for the attribute\n        '
        index = self.domain.index(attribute)
        if index < 0:
            index = len(self.domain.variables) - 1 - index
        index = self.index(index, 1)
        self.dataChanged.emit(index, index)

    def rowCount(self, parent=None) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 0 if self.domain is None or (parent is not None and parent.isValid()) else len(self.domain.variables) + len(self.domain.metas)

    @staticmethod
    def columnCount(parent=None) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 0 if parent is not None and parent.isValid() else len(TABLE_COLUMN_NAMES)

    def data(self, index, role=Qt.DisplayRole) -> Any:
        if False:
            i = 10
            return i + 15
        (row, col) = (index.row(), index.column())
        val = (self.domain.variables + self.domain.metas)[row]
        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == TabColumn.attribute:
                return str(val)
            else:
                aggs = sorted(self.parent.aggregations.get(val, []), key=AGGREGATIONS_ORD.index)
                n_more = '' if len(aggs) <= 3 else f' and {len(aggs) - 3} more'
                return ', '.join(aggs[:3]) + n_more
        elif role == Qt.DecorationRole and col == TabColumn.attribute:
            return gui.attributeIconDict[val]
        return None

    def headerData(self, i, orientation, role=Qt.DisplayRole) -> str:
        if False:
            while True:
                i = 10
        if orientation == Qt.Horizontal and role == Qt.DisplayRole and (i < 2):
            return TABLE_COLUMN_NAMES[i]
        return super().headerData(i, orientation, role)

class AggregateListViewSearch(ListViewFilter):
    """ListViewSearch that disables unselecting all items in the list"""

    def selectionCommand(self, index: QModelIndex, event: QEvent=None) -> QItemSelectionModel.SelectionFlags:
        if False:
            for i in range(10):
                print('nop')
        flags = super().selectionCommand(index, event)
        selmodel = self.selectionModel()
        if not index.isValid():
            return QItemSelectionModel.NoUpdate
        if selmodel.isSelected(index):
            currsel = selmodel.selectedIndexes()
            if len(currsel) == 1 and index == currsel[0]:
                return QItemSelectionModel.NoUpdate
        if event is not None and event.type() == QEvent.MouseMove and flags & QItemSelectionModel.ToggleCurrent:
            flags &= ~QItemSelectionModel.Toggle
            flags |= QItemSelectionModel.Select
        return flags

class CheckBox(QCheckBox):

    def __init__(self, text, parent):
        if False:
            i = 10
            return i + 15
        super().__init__(text)
        self.parent: OWGroupBy = parent

    def nextCheckState(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Custom behaviour for switching between steps. It is required since\n        sometimes user will select different types of attributes at the same\n        time. In this case we step between unchecked, partially checked and\n        checked or just between unchecked and checked - depending on situation\n        '
        if self.checkState() == Qt.Checked:
            self.setCheckState(Qt.Unchecked)
        else:
            agg = self.text()
            selected_attrs = self.parent.get_selected_attributes()
            types = set((type(attr) for attr in selected_attrs))
            can_be_applied_all = types <= AGGREGATIONS[agg].types
            applied_all = all((type(attr) not in AGGREGATIONS[agg].types or agg in self.parent.aggregations[attr] for attr in selected_attrs))
            if self.checkState() == Qt.PartiallyChecked:
                if can_be_applied_all:
                    self.setCheckState(Qt.Checked)
                elif applied_all:
                    self.setCheckState(Qt.Unchecked)
                else:
                    self.setCheckState(Qt.PartiallyChecked)
                    self.stateChanged.emit(enum_as_int(Qt.PartiallyChecked))
            else:
                self.setCheckState(Qt.Checked if can_be_applied_all else Qt.PartiallyChecked)

@contextmanager
def block_signals(widget):
    if False:
        while True:
            i = 10
    widget.blockSignals(True)
    try:
        yield
    finally:
        widget.blockSignals(False)

class OWGroupBy(OWWidget, ConcurrentWidgetMixin):
    name = 'Group by'
    description = ''
    category = 'Transform'
    icon = 'icons/GroupBy.svg'
    keywords = 'aggregate, group by'
    priority = 1210

    class Inputs:
        data = Input('Data', Table, doc='Input data table')

    class Outputs:
        data = Output('Data', Table, doc='Aggregated data')

    class Error(OWWidget.Error):
        unexpected_error = Msg('{}')
    settingsHandler = DomainContextHandler()
    gb_attrs: List[Variable] = ContextSetting([])
    aggregations: Dict[Variable, Set[str]] = ContextSetting({})
    auto_commit: bool = Setting(True)

    def __init__(self):
        if False:
            return 10
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.data = None
        self.result = None
        self.gb_attrs_model = DomainModel(separators=False)
        self.agg_table_model = VarTableModel(self)
        self.agg_checkboxes = {}
        self.__init_control_area()
        self.__init_main_area()

    def __init_control_area(self) -> None:
        if False:
            return 10
        'Init all controls in the control area'
        gui.listView(self.controlArea, self, 'gb_attrs', box='Group by', model=self.gb_attrs_model, viewType=AggregateListViewSearch, callback=self.__gb_changed, selectionMode=ListViewFilter.ExtendedSelection)
        gui.auto_send(self.buttonsArea, self, 'auto_commit')

    def __init_main_area(self) -> None:
        if False:
            return 10
        'Init all controls in the main area'
        self.agg_table_view = tableview = QTableView()
        tableview.setModel(self.agg_table_model)
        tableview.setSelectionBehavior(QAbstractItemView.SelectRows)
        tableview.selectionModel().selectionChanged.connect(self.__rows_selected)
        tableview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        vbox = gui.vBox(self.mainArea, ' ')
        vbox.layout().addWidget(tableview)
        grid_layout = QGridLayout()
        gui.widgetBox(self.mainArea, orientation=grid_layout, box='Aggregations')
        col = 0
        row = 0
        break_rows = (6, 6, 99)
        for agg in AGGREGATIONS:
            self.agg_checkboxes[agg] = cb = CheckBox(agg, self)
            cb.setDisabled(True)
            cb.stateChanged.connect(partial(self.__aggregation_changed, agg))
            grid_layout.addWidget(cb, row, col)
            row += 1
            if row == break_rows[col]:
                row = 0
                col += 1

    def __rows_selected(self) -> None:
        if False:
            print('Hello World!')
        'Callback for table selection change; update checkboxes'
        selected_attrs = self.get_selected_attributes()
        types = {type(attr) for attr in selected_attrs}
        active_aggregations = [self.aggregations[attr] for attr in selected_attrs]
        for (agg, cb) in self.agg_checkboxes.items():
            cb.setDisabled(not types & AGGREGATIONS[agg].types)
            activated = {agg in a for a in active_aggregations}
            with block_signals(cb):
                cb.setCheckState(Qt.Checked if activated == {True} else Qt.Unchecked if activated == {False} else Qt.PartiallyChecked)

    def __gb_changed(self) -> None:
        if False:
            i = 10
            return i + 15
        'Callback for Group-by attributes selection change'
        self.result = Result()
        self.commit.deferred()

    def __aggregation_changed(self, agg: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Callback for aggregation change; update aggregations dictionary and call\n        commit\n        '
        selected_attrs = self.get_selected_attributes()
        for attr in selected_attrs:
            if self.agg_checkboxes[agg].isChecked() and self.__aggregation_compatible(agg, attr):
                self.aggregations[attr].add(agg)
            else:
                self.aggregations[attr].discard(agg)
            self.agg_table_model.update_aggregation(attr)
        self.commit.deferred()

    @Inputs.data
    def set_data(self, data: Table) -> None:
        if False:
            while True:
                i = 10
        self.closeContext()
        self.data = data
        self.cancel()
        self.result = Result()
        self.Outputs.data.send(None)
        self.gb_attrs_model.set_domain(data.domain if data else None)
        self.gb_attrs = self.gb_attrs_model[:1] if self.gb_attrs_model else []
        self.aggregations = {attr: DEFAULT_AGGREGATIONS[type(attr)].copy() for attr in data.domain.variables + data.domain.metas} if data else {}
        default_aggregations = self.aggregations.copy()
        self.openContext(self.data)
        self.aggregations.update({k: v for (k, v) in default_aggregations.items() if k not in self.aggregations})
        self.agg_table_model.set_domain(data.domain if data else None)
        self._set_gb_selection()
        self.commit.now()

    @gui.deferred
    def commit(self) -> None:
        if False:
            return 10
        self.Error.clear()
        self.Warning.clear()
        if self.data:
            self.start(_run, self.data, self.gb_attrs, self.aggregations, self.result)

    def on_done(self, result: Result) -> None:
        if False:
            print('Hello World!')
        self.result = result
        self.Outputs.data.send(result.result_table)

    def on_partial_result(self, result: Result) -> None:
        if False:
            return 10
        self.result = result

    def on_exception(self, ex: Exception):
        if False:
            i = 10
            return i + 15
        self.Error.unexpected_error(str(ex))

    def get_selected_attributes(self):
        if False:
            return 10
        'Get select attributes in the table'
        selection_model = self.agg_table_view.selectionModel()
        sel_rows = selection_model.selectedRows()
        vars_ = self.data.domain.variables + self.data.domain.metas
        return [vars_[index.row()] for index in sel_rows]

    def _set_gb_selection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update selected attributes. When context includes variable hidden in\n        data, it will match and gb_attrs may include hidden attribute. Remove it\n        since otherwise widget groups by attribute that is not present in view.\n        '
        values = self.gb_attrs_model[:]
        self.gb_attrs = [var_ for var_ in self.gb_attrs if var_ in values]
        if not self.gb_attrs and self.gb_attrs_model:
            self.gb_attrs = self.gb_attrs_model[:1]

    @staticmethod
    def __aggregation_compatible(agg, attr):
        if False:
            return 10
        'Check a compatibility of aggregation with the variable'
        return type(attr) in AGGREGATIONS[agg].types

    @classmethod
    def migrate_context(cls, context, _):
        if False:
            return 10
        '\n        Before widget allowed using Sum on Time variable, now it is forbidden.\n        This function removes Sum from the context for TimeVariables (104)\n        '
        for (var_, v) in context.values['aggregations'][0].items():
            if len(var_) == 2:
                if var_[1] == 104:
                    v.discard('Sum')
if __name__ == '__main__':
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWGroupBy).run(Table('iris'))