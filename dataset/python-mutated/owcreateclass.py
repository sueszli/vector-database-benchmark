"""Widget for creating classes from non-numeric attribute by substrings"""
import re
from itertools import count
import numpy as np
from AnyQt.QtWidgets import QGridLayout, QLabel, QLineEdit, QSizePolicy, QWidget
from AnyQt.QtCore import Qt
from Orange.data import StringVariable, DiscreteVariable, Domain
from Orange.data.table import Table
from Orange.statistics.util import bincount
from Orange.preprocess.transformation import Transformation, Lookup
from Orange.widgets import gui, widget
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.localization import pl
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, Input, Output

def map_by_substring(a, patterns, case_sensitive, match_beginning, map_values=None):
    if False:
        return 10
    '\n    Map values in a using a list of patterns. The patterns are considered in\n    order of appearance.\n\n    Args:\n        a (np.array): input array of `dtype` `str`\n        patterns (list of str): list of strings\n        case_sensitive (bool): case sensitive match\n        match_beginning (bool): match only at the beginning of the string\n        map_values (list of int): list of len(pattens);\n                                  contains return values for each pattern\n\n    Returns:\n        np.array of floats representing indices of matched patterns\n    '
    if map_values is None:
        map_values = np.arange(len(patterns))
    else:
        map_values = np.array(map_values, dtype=int)
    res = np.full(len(a), np.nan)
    if not case_sensitive:
        a = np.char.lower(a)
        patterns = (pattern.lower() for pattern in patterns)
    for (val_idx, pattern) in reversed(list(enumerate(patterns))):
        indices = np.char.find(a, pattern)
        matches = indices == 0 if match_beginning else indices != -1
        res[matches] = map_values[val_idx]
    return res

class ValueFromStringSubstring(Transformation):
    """
    Transformation that computes a discrete variable from a string variable by
    pattern matching.

    Given patterns `["abc", "a", "bc", ""]`, string data
    `["abcd", "aa", "bcd", "rabc", "x"]` is transformed to values of the new
    attribute with indices`[0, 1, 2, 0, 3]`.

    Args:
        variable (:obj:`~Orange.data.StringVariable`): the original variable
        patterns (list of str): list of string patterns
        case_sensitive (bool, optional): if set to `True`, the match is case
            sensitive
        match_beginning (bool, optional): if set to `True`, the pattern must
            appear at the beginning of the string
    """

    def __init__(self, variable, patterns, case_sensitive=False, match_beginning=False, map_values=None):
        if False:
            i = 10
            return i + 15
        super().__init__(variable)
        self.patterns = patterns
        self.case_sensitive = case_sensitive
        self.match_beginning = match_beginning
        self.map_values = map_values

    def transform(self, c):
        if False:
            print('Hello World!')
        '\n        Transform the given data.\n\n        Args:\n            c (np.array): an array of type that can be cast to dtype `str`\n\n        Returns:\n            np.array of floats representing indices of matched patterns\n        '
        nans = np.equal(c, None)
        c = c.astype(str)
        c[nans] = ''
        res = map_by_substring(c, self.patterns, self.case_sensitive, self.match_beginning, self.map_values)
        res[nans] = np.nan
        return res

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return super().__eq__(other) and self.patterns == other.patterns and (self.case_sensitive == other.case_sensitive) and (self.match_beginning == other.match_beginning) and (self.map_values == other.map_values)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((type(self), self.variable, tuple(self.patterns), self.case_sensitive, self.match_beginning, self.map_values))

class ValueFromDiscreteSubstring(Lookup):
    """
    Transformation that computes a discrete variable from discrete variable by
    pattern matching.

    Say that the original attribute has values
    `["abcd", "aa", "bcd", "rabc", "x"]`. Given patterns
    `["abc", "a", "bc", ""]`, the values are mapped to the values of the new
    attribute with indices`[0, 1, 2, 0, 3]`.

    Args:
        variable (:obj:`~Orange.data.DiscreteVariable`): the original variable
        patterns (list of str): list of string patterns
        case_sensitive (bool, optional): if set to `True`, the match is case
            sensitive
        match_beginning (bool, optional): if set to `True`, the pattern must
            appear at the beginning of the string
    """

    def __init__(self, variable, patterns, case_sensitive=False, match_beginning=False, map_values=None):
        if False:
            while True:
                i = 10
        super().__init__(variable, [])
        self.case_sensitive = case_sensitive
        self.match_beginning = match_beginning
        self.map_values = map_values
        self.patterns = patterns

    def __setattr__(self, key, value):
        if False:
            i = 10
            return i + 15
        '__setattr__ is overloaded to recompute the lookup table when the\n        patterns, the original attribute or the flags change.'
        super().__setattr__(key, value)
        if hasattr(self, 'patterns') and key in ('case_sensitive', 'match_beginning', 'patterns', 'variable', 'map_values'):
            self.lookup_table = map_by_substring(self.variable.values, self.patterns, self.case_sensitive, self.match_beginning, self.map_values)

def unique_in_order_mapping(a):
    if False:
        while True:
            i = 10
    ' Return\n    - unique elements of the input list (in the order of appearance)\n    - indices of the input list onto the returned uniques\n    '
    first_position = {}
    unique_in_order = []
    mapping = []
    for e in a:
        if e not in first_position:
            first_position[e] = len(unique_in_order)
            unique_in_order.append(e)
        mapping.append(first_position[e])
    return (unique_in_order, mapping)

class OWCreateClass(widget.OWWidget):
    name = 'Create Class'
    description = 'Create class attribute from a string attribute'
    icon = 'icons/CreateClass.svg'
    category = 'Transform'
    keywords = 'create class'
    priority = 2300

    class Inputs:
        data = Input('Data', Table)

    class Outputs:
        data = Output('Data', Table)
    want_main_area = False
    buttons_area_orientation = Qt.Vertical
    settingsHandler = DomainContextHandler()
    attribute = ContextSetting(None)
    class_name = ContextSetting('class')
    rules = ContextSetting({})
    match_beginning = ContextSetting(False)
    case_sensitive = ContextSetting(False)
    TRANSFORMERS = {StringVariable: ValueFromStringSubstring, DiscreteVariable: ValueFromDiscreteSubstring}
    cached_variables = {}

    class Warning(widget.OWWidget.Warning):
        no_nonnumeric_vars = Msg('Data contains only numeric variables.')

    class Error(widget.OWWidget.Error):
        class_name_duplicated = Msg('Class name duplicated.')
        class_name_empty = Msg('Class name should not be empty.')

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.data = None
        self.match_counts = []
        self.line_edits = []
        self.remove_buttons = []
        self.counts = []
        gui.lineEdit(self.controlArea, self, 'class_name', orientation=Qt.Horizontal, box='New Class Name')
        variable_select_box = gui.vBox(self.controlArea, 'Match by Substring')
        combo = gui.comboBox(variable_select_box, self, 'attribute', label='From column:', orientation=Qt.Horizontal, searchable=True, callback=self.update_rules, model=DomainModel(valid_types=(StringVariable, DiscreteVariable)))
        combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        patternbox = gui.vBox(variable_select_box)
        self.rules_box = rules_box = QGridLayout()
        rules_box.setSpacing(4)
        rules_box.setContentsMargins(4, 4, 4, 4)
        self.rules_box.setColumnMinimumWidth(1, 70)
        self.rules_box.setColumnMinimumWidth(0, 10)
        self.rules_box.setColumnStretch(0, 1)
        self.rules_box.setColumnStretch(1, 1)
        self.rules_box.setColumnStretch(2, 100)
        rules_box.addWidget(QLabel('Name'), 0, 1)
        rules_box.addWidget(QLabel('Substring'), 0, 2)
        rules_box.addWidget(QLabel('Count'), 0, 3, 1, 2)
        self.update_rules()
        widget = QWidget(patternbox)
        widget.setLayout(rules_box)
        patternbox.layout().addWidget(widget)
        box = gui.hBox(patternbox)
        gui.rubber(box)
        gui.button(box, self, '+', callback=self.add_row, autoDefault=False, width=34, sizePolicy=(QSizePolicy.Maximum, QSizePolicy.Maximum))
        optionsbox = gui.vBox(self.controlArea, 'Options')
        gui.checkBox(optionsbox, self, 'match_beginning', 'Match only at the beginning', callback=self.options_changed)
        gui.checkBox(optionsbox, self, 'case_sensitive', 'Case sensitive', callback=self.options_changed)
        gui.rubber(self.controlArea)
        gui.button(self.buttonsArea, self, 'Apply', callback=self.apply)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    @property
    def active_rules(self):
        if False:
            while True:
                i = 10
        '\n        Returns the class names and patterns corresponding to the currently\n            selected attribute. If the attribute is not yet in the dictionary,\n            set the default.\n        '
        return self.rules.setdefault(self.attribute and self.attribute.name, [['', ''], ['', '']])

    def rules_to_edits(self):
        if False:
            while True:
                i = 10
        'Fill the line edites with the rules from the current settings.'
        for (editr, textr) in zip(self.line_edits, self.active_rules):
            for (edit, text) in zip(editr, textr):
                edit.setText(text)

    @Inputs.data
    def set_data(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Input data signal handler.'
        self.closeContext()
        self.rules = {}
        self.data = data
        model = self.controls.attribute.model()
        model.set_domain(data.domain if data is not None else None)
        self.Warning.no_nonnumeric_vars(shown=data is not None and (not model))
        if not model:
            self.attribute = None
            self.Outputs.data.send(None)
            return
        self.attribute = model[0]
        self.openContext(data)
        self.update_rules()
        self.apply()

    def update_rules(self):
        if False:
            while True:
                i = 10
        'Called when the rules are changed: adjust the number of lines in\n        the form and fill them, update the counts. The widget does not have\n        auto-apply.'
        self.adjust_n_rule_rows()
        self.rules_to_edits()
        self.update_counts()

    def options_changed(self):
        if False:
            print('Hello World!')
        self.update_counts()

    def adjust_n_rule_rows(self):
        if False:
            while True:
                i = 10
        'Add or remove lines if needed and fix the tab order.'

        def _add_line():
            if False:
                return 10
            self.line_edits.append([])
            n_lines = len(self.line_edits)
            for coli in range(1, 3):
                edit = QLineEdit()
                self.line_edits[-1].append(edit)
                self.rules_box.addWidget(edit, n_lines, coli)
                edit.textChanged.connect(self.sync_edit)
            button = gui.button(None, self, label='×', width=33, autoDefault=False, callback=self.remove_row, sizePolicy=(QSizePolicy.Maximum, QSizePolicy.Maximum))
            self.remove_buttons.append(button)
            self.rules_box.addWidget(button, n_lines, 0)
            self.counts.append([])
            for (coli, kwargs) in enumerate((dict(), dict(styleSheet='color: gray'))):
                label = QLabel(alignment=Qt.AlignCenter, **kwargs)
                self.counts[-1].append(label)
                self.rules_box.addWidget(label, n_lines, 3 + coli)

        def _remove_line():
            if False:
                return 10
            for edit in self.line_edits.pop():
                edit.deleteLater()
            self.remove_buttons.pop().deleteLater()
            for label in self.counts.pop():
                label.deleteLater()

        def _fix_tab_order():
            if False:
                for i in range(10):
                    print('nop')
            prev = None
            for (row, rule) in zip(self.line_edits, self.active_rules):
                for (col_idx, edit) in enumerate(row):
                    (edit.row, edit.col_idx) = (rule, col_idx)
                    if prev is not None:
                        self.setTabOrder(prev, edit)
                    prev = edit
        n = len(self.active_rules)
        while n > len(self.line_edits):
            _add_line()
        while len(self.line_edits) > n:
            _remove_line()
        _fix_tab_order()

    def add_row(self):
        if False:
            while True:
                i = 10
        'Append a new row at the end.'
        self.active_rules.append(['', ''])
        self.adjust_n_rule_rows()
        self.update_counts()

    def remove_row(self):
        if False:
            for i in range(10):
                print('nop')
        'Remove a row.'
        remove_idx = self.remove_buttons.index(self.sender())
        del self.active_rules[remove_idx]
        self.update_rules()
        self.update_counts()

    def sync_edit(self, text):
        if False:
            while True:
                i = 10
        'Handle changes in line edits: update the active rules and counts'
        edit = self.sender()
        edit.row[edit.col_idx] = text
        self.update_counts()

    def class_labels(self):
        if False:
            print('Hello World!')
        'Construct a list of class labels. Empty labels are replaced with\n        C1, C2, C3. If C<n> already appears in the list of values given by\n        the user, the labels start at C<n+1> instead.\n        '
        largest_c = max((int(label[1:]) for (label, _) in self.active_rules if re.match('^C\\d+', label)), default=0)
        class_count = count(largest_c + 1)
        return [label_edit.text() or 'C{}'.format(next(class_count)) for (label_edit, _) in self.line_edits]

    def update_counts(self):
        if False:
            return 10
        'Recompute and update the counts of matches.'

        def _matcher(strings, pattern):
            if False:
                return 10
            'Return indices of strings into patterns; consider case\n            sensitivity and matching at the beginning. The given strings are\n            assumed to be in lower case if match is case insensitive. Patterns\n            are fixed on the fly.'
            if not self.case_sensitive:
                pattern = pattern.lower()
            indices = np.char.find(strings, pattern.strip())
            return indices == 0 if self.match_beginning else indices != -1

        def _lower_if_needed(strings):
            if False:
                i = 10
                return i + 15
            return strings if self.case_sensitive else np.char.lower(strings)

        def _string_counts():
            if False:
                i = 10
                return i + 15
            '\n            Generate pairs of arrays for each rule until running out of data\n            instances. np.sum over the two arrays in each pair gives the\n            number of matches of the remaining instances (considering the\n            order of patterns) and of the original data.\n\n            For _string_counts, the arrays contain bool masks referring to the\n            original data\n            '
            nonlocal data
            data = data.astype(str)
            data = data[~np.char.equal(data, '')]
            data = _lower_if_needed(data)
            remaining = np.array(data)
            for (_, pattern) in self.active_rules:
                matching = _matcher(remaining, pattern)
                total_matching = _matcher(data, pattern)
                yield (matching, total_matching)
                remaining = remaining[~matching]
                if not remaining.size:
                    break

        def _discrete_counts():
            if False:
                for i in range(10):
                    print('nop')
            "\n            Generate pairs similar to _string_counts, except that the arrays\n            contain bin counts for the attribute's values matching the pattern.\n            "
            attr_vals = np.array(attr.values)
            attr_vals = _lower_if_needed(attr_vals)
            bins = bincount(data, max_val=len(attr.values) - 1)[0]
            remaining = np.array(bins)
            for (_, pattern) in self.active_rules:
                matching = _matcher(attr_vals, pattern)
                yield (remaining[matching], bins[matching])
                remaining[matching] = 0
                if not np.any(remaining):
                    break

        def _clear_labels():
            if False:
                return 10
            'Clear all labels'
            for (lab_matched, lab_total) in self.counts:
                lab_matched.setText('')
                lab_total.setText('')

        def _set_labels():
            if False:
                print('Hello World!')
            'Set the labels to show the counts'
            for ((n_matched, n_total), (lab_matched, lab_total), (lab, patt)) in zip(self.match_counts, self.counts, self.active_rules):
                n_before = n_total - n_matched
                lab_matched.setText('{}'.format(n_matched))
                if n_before and (lab or patt):
                    lab_total.setText('+ {}'.format(n_before))
                    if n_matched:
                        tip = f"{n_before} of {n_total} matching {pl(n_total, 'instance')} {pl(n_before, 'is|are')} already covered above."
                    else:
                        tip = 'All matching instances are already covered above'
                    lab_total.setToolTip(tip)
                    lab_matched.setToolTip(tip)

        def _set_placeholders():
            if False:
                for i in range(10):
                    print('nop')
            'Set placeholders for empty edit lines'
            matches = [n for (n, _) in self.match_counts] + [0] * len(self.line_edits)
            for (n_matched, (_, patt)) in zip(matches, self.line_edits):
                if not patt.text():
                    patt.setPlaceholderText('(remaining instances)' if n_matched else '(unused)')
            labels = self.class_labels()
            for (label, (lab_edit, _)) in zip(labels, self.line_edits):
                if not lab_edit.text():
                    lab_edit.setPlaceholderText(label)
        _clear_labels()
        attr = self.attribute
        if attr is None:
            return
        counters = {StringVariable: _string_counts, DiscreteVariable: _discrete_counts}
        data = self.data.get_column(attr)
        self.match_counts = [[int(np.sum(x)) for x in matches] for matches in counters[type(attr)]()]
        _set_labels()
        _set_placeholders()

    def apply(self):
        if False:
            for i in range(10):
                print('nop')
        'Output the transformed data.'
        self.Error.clear()
        self.class_name = self.class_name.strip()
        if not self.attribute:
            self.Outputs.data.send(None)
            return
        domain = self.data.domain
        if not self.class_name:
            self.Error.class_name_empty()
        if self.class_name in domain:
            self.Error.class_name_duplicated()
        if not self.class_name or self.class_name in domain:
            self.Outputs.data.send(None)
            return
        new_class = self._create_variable()
        new_domain = Domain(domain.attributes, new_class, domain.metas + domain.class_vars)
        new_data = self.data.transform(new_domain)
        self.Outputs.data.send(new_data)

    def _create_variable(self):
        if False:
            i = 10
            return i + 15
        rules = self.active_rules
        valid_rules = [label or pattern or n_matches for ((label, pattern), n_matches) in zip(rules, self.match_counts)]
        patterns = tuple((pattern for ((_, pattern), valid) in zip(rules, valid_rules) if valid))
        names = tuple((name for (name, valid) in zip(self.class_labels(), valid_rules) if valid))
        transformer = self.TRANSFORMERS[type(self.attribute)]
        (names, map_values) = unique_in_order_mapping(names)
        names = tuple((str(a) for a in names))
        map_values = tuple(map_values)
        var_key = (self.attribute, self.class_name, names, patterns, self.case_sensitive, self.match_beginning, map_values)
        if var_key in self.cached_variables:
            return self.cached_variables[var_key]
        compute_value = transformer(self.attribute, patterns, self.case_sensitive, self.match_beginning, map_values)
        new_var = DiscreteVariable(self.class_name, names, compute_value=compute_value)
        self.cached_variables[var_key] = new_var
        return new_var

    def send_report(self):
        if False:
            return 10

        def _cond_part():
            if False:
                for i in range(10):
                    print('nop')
            rule = f'<b>{class_name}</b> '
            if patt:
                rule += f'if <b>{self.attribute.name}</b> contains <b>{patt}</b>'
            else:
                rule += 'otherwise'
            return rule

        def _count_part():
            if False:
                print('Hello World!')
            aca = 'already covered above'
            if not n_matched:
                if n_total == 1:
                    return f'the single matching instance is {aca}'
                elif n_total == 2:
                    return f'both matching instances are {aca}'
                else:
                    return f'all {n_total} matching instances are {aca}'
            elif not patt:
                return f"{n_matched} {pl(n_matched, 'instance')}"
            else:
                m = f"{n_matched} matching {pl(n_matched, 'instance')}"
                if n_matched < n_total:
                    n_already = n_total - n_matched
                    m += f" (+{n_already} that {pl(n_already, 'is|are')} {aca})"
                return m
        if not self.attribute:
            return
        self.report_items('Input', [('Source attribute', self.attribute.name)])
        output = ''
        names = self.class_labels()
        for ((n_matched, n_total), class_name, (lab, patt)) in zip(self.match_counts, names, self.active_rules):
            if lab or patt or n_total:
                output += '<li>{}; {}</li>'.format(_cond_part(), _count_part())
        if output:
            self.report_items('Output', [('Class name', self.class_name)])
            self.report_raw('<ol>{}</ol>'.format(output))
if __name__ == '__main__':
    WidgetPreview(OWCreateClass).run(Table('zoo'))