from os import path, remove, getcwd
from os.path import dirname
import unittest
from threading import Thread
from unittest.mock import Mock, patch
import pickle
import tempfile
import warnings
import numpy as np
import scipy.sparse as sp
from AnyQt.QtCore import QMimeData, QPoint, Qt, QUrl, QPointF
from AnyQt.QtGui import QDragEnterEvent, QDropEvent
from AnyQt.QtTest import QTest
from AnyQt.QtWidgets import QComboBox
import Orange
from Orange.data import FileFormat, dataset_dirs, StringVariable, Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.util import OrangeDeprecationWarning
from Orange.data.io import TabReader
from Orange.tests import named_file
from Orange.widgets.data.owfile import OWFile, OWFileDropHandler, DEFAULT_READER_TEXT
from Orange.widgets.utils.filedialogs import dialog_formats, format_filter, RecentPath
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.domaineditor import ComboDelegate, VarTypeDelegate, VarTableModel
TITANIC_PATH = path.join(path.dirname(Orange.__file__), 'datasets', 'titanic.tab')
orig_path_exists = path.exists

class FailedSheetsFormat(FileFormat):
    EXTENSIONS = ('.failed_sheet',)
    DESCRIPTION = 'Make a sheet function that fails'

    def read(self):
        if False:
            return 10
        pass

    def sheets(self):
        if False:
            while True:
                i = 10
        raise Exception('Not working')

class WithWarnings(FileFormat):
    EXTENSIONS = ('.with_warning',)
    DESCRIPTION = 'Warning'

    @staticmethod
    def read():
        if False:
            i = 10
            return i + 15
        warnings.warn('Some warning')
        return Orange.data.Table('iris')

class MyCustomTabReader(FileFormat):
    EXTENSIONS = ('.tab',)
    DESCRIPTION = 'Always return iris'
    PRIORITY = 999999

    @staticmethod
    def read():
        if False:
            for i in range(10):
                print('nop')
        return Orange.data.Table('iris')

class TestOWFile(WidgetTest):
    event_data = None

    def setUp(self):
        if False:
            print('Hello World!')
        self.widget = self.create_widget(OWFile)
        dataset_dirs.append(dirname(__file__))

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        dataset_dirs.pop()

    def test_describe_call_get_nans(self):
        if False:
            print('Hello World!')
        table = Table('iris')
        with patch.object(Table, 'get_nan_frequency_attribute', return_value=0.0) as mock:
            self.widget._describe(table)
            mock.assert_called()
        table = Table.from_numpy(domain=None, X=np.random.random((10000, 1000)))
        with patch.object(Table, 'get_nan_frequency_attribute', return_value=0.0) as mock:
            self.widget._describe(table)
            mock.assert_not_called()

    def test_dragEnterEvent_accepts_urls(self):
        if False:
            i = 10
            return i + 15
        event = self._drag_enter_event(QUrl.fromLocalFile(TITANIC_PATH))
        self.widget.dragEnterEvent(event)
        self.assertTrue(event.isAccepted())

    def test_dragEnterEvent_skips_osx_file_references(self):
        if False:
            while True:
                i = 10
        event = self._drag_enter_event(QUrl.fromLocalFile('/.file/id=12345'))
        self.widget.dragEnterEvent(event)
        self.assertFalse(event.isAccepted())

    def test_dragEnterEvent_skips_usupported_files(self):
        if False:
            print('Hello World!')
        event = self._drag_enter_event(QUrl.fromLocalFile('file.unsupported'))
        self.widget.dragEnterEvent(event)
        self.assertFalse(event.isAccepted())

    def _drag_enter_event(self, url):
        if False:
            print('Hello World!')
        self.event_data = data = QMimeData()
        data.setUrls([QUrl(url)])
        return QDragEnterEvent(QPoint(0, 0), Qt.MoveAction, data, Qt.NoButton, Qt.NoModifier)

    def test_dropEvent_selects_file(self):
        if False:
            for i in range(10):
                print('nop')
        self.widget.load_data = Mock()
        self.widget.source = OWFile.URL
        event = self._drop_event(QUrl.fromLocalFile(TITANIC_PATH))
        self.widget.dropEvent(event)
        self.assertEqual(self.widget.source, OWFile.LOCAL_FILE)
        self.assertTrue(path.samefile(self.widget.last_path(), TITANIC_PATH))
        self.widget.load_data.assert_called_with()

    def _drop_event(self, url):
        if False:
            for i in range(10):
                print('nop')
        self.event_data = data = QMimeData()
        data.setUrls([QUrl(url)])
        return QDropEvent(QPointF(0, 0), Qt.MoveAction, data, Qt.NoButton, Qt.NoModifier, QDropEvent.Drop)

    def test_check_file_size(self):
        if False:
            print('Hello World!')
        self.assertFalse(self.widget.Warning.file_too_big.is_shown())
        self.widget.SIZE_LIMIT = 4000
        self.widget.__init__()
        self.assertTrue(self.widget.Warning.file_too_big.is_shown())

    def test_domain_changes_are_stored(self):
        if False:
            return 10
        assert isinstance(self.widget, OWFile)
        self.open_dataset('iris')
        idx = self.widget.domain_editor.model().createIndex(4, 1)
        self.widget.domain_editor.model().setData(idx, 'text', Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIsInstance(data.domain['iris'], StringVariable)
        self.open_dataset('zoo')
        data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(data.name, 'zoo')
        self.open_dataset('iris')
        data = self.get_output(self.widget.Outputs.data)
        self.assertIsInstance(data.domain['iris'], StringVariable)

    def test_rename_duplicates(self):
        if False:
            print('Hello World!')
        self.open_dataset('iris')
        idx = self.widget.domain_editor.model().createIndex(3, 0)
        self.assertFalse(self.widget.Warning.renamed_vars.is_shown())
        self.widget.domain_editor.model().setData(idx, 'iris', Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn('iris (1)', data.domain)
        self.assertIn('iris (2)', data.domain)
        self.assertTrue(self.widget.Warning.renamed_vars.is_shown())
        self.widget.domain_editor.model().setData(idx, 'different iris', Qt.EditRole)
        self.widget.apply_button.click()
        self.assertFalse(self.widget.Warning.renamed_vars.is_shown())

    def test_variable_name_change(self):
        if False:
            print('Hello World!')
        '\n        Test whether the name of the variable is changed correctly by\n        the domaineditor.\n        '
        self.open_dataset('iris')
        idx = self.widget.domain_editor.model().createIndex(4, 0)
        self.widget.domain_editor.model().setData(idx, 'a', Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn('a', data.domain)
        idx = self.widget.domain_editor.model().createIndex(3, 0)
        self.widget.domain_editor.model().setData(idx, 'd', Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn('d', data.domain)
        idx = self.widget.domain_editor.model().createIndex(4, 0)
        self.widget.domain_editor.model().setData(idx, 'b', Qt.EditRole)
        idx = self.widget.domain_editor.model().createIndex(4, 1)
        self.widget.domain_editor.model().setData(idx, 'text', Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn('b', data.domain)
        self.assertIsInstance(data.domain['b'], StringVariable)
        idx = self.widget.domain_editor.model().createIndex(4, 0)
        self.widget.domain_editor.model().setData(idx, 'c', Qt.EditRole)
        idx = self.widget.domain_editor.model().createIndex(4, 1)
        self.widget.domain_editor.model().setData(idx, 'categorical', Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn('c', data.domain)
        self.assertIsInstance(data.domain['c'], DiscreteVariable)
        self.open_dataset('zoo')
        idx = self.widget.domain_editor.model().createIndex(0, 0)
        self.widget.domain_editor.model().setData(idx, 'c', Qt.EditRole)
        idx = self.widget.domain_editor.model().createIndex(0, 1)
        self.widget.domain_editor.model().setData(idx, 'numeric', Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn('c', data.domain)
        self.assertIsInstance(data.domain['c'], ContinuousVariable)

    def open_dataset(self, name):
        if False:
            i = 10
            return i + 15
        filename = FileFormat.locate(name, dataset_dirs)
        self.widget.add_path(filename)
        self.widget.load_data()

    def test_no_last_path(self):
        if False:
            while True:
                i = 10
        self.widget = self.create_widget(OWFile, stored_settings={'recent_paths': []})
        self.assertEqual(self.widget.file_combo.count(), 1)

    def test_file_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        file_name = 'test_owfile_data.tab'
        domainA = Domain([DiscreteVariable('d1', values=('a', 'b'))], DiscreteVariable('c1', values=('aaa', 'bbb')))
        dataA = Table(domainA, np.array([[0], [1], [0], [np.nan]]), np.array([0, 1, 0, 1]))
        dataA.save(file_name)
        self.open_dataset(file_name)
        self.assertEqual(self.get_output(self.widget.Outputs.data).domain, dataA.domain)
        remove(file_name)
        self.widget.load_data()
        self.assertEqual(file_name, path.basename(self.widget.last_path()))
        self.assertTrue(self.widget.Error.file_not_found.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertEqual(self.widget.infolabel.text(), 'No data.')
        self.open_dataset('iris')
        self.assertFalse(self.widget.Error.file_not_found.is_shown())

    def test_nothing_selected(self):
        if False:
            while True:
                i = 10
        widget = self.widget = self.create_widget(OWFile, stored_settings={'recent_paths': []})
        widget.Outputs.data.send = Mock()
        widget.load_data()
        self.assertTrue(widget.Information.no_file_selected.is_shown())
        widget.Outputs.data.send.assert_called_with(None)
        widget.Outputs.data.send.reset_mock()
        widget.source = widget.URL
        widget.load_data()
        self.assertTrue(widget.Information.no_file_selected.is_shown())
        widget.Outputs.data.send.assert_called_with(None)

    def test_check_column_noname(self):
        if False:
            return 10
        '\n        Column name cannot be changed to an empty string or a string with whitespaces.\n        GH-2039\n        '
        self.open_dataset('iris')
        idx = self.widget.domain_editor.model().createIndex(1, 0)
        temp = self.widget.domain_editor.model().data(idx, Qt.DisplayRole)
        self.widget.domain_editor.model().setData(idx, '   ', Qt.EditRole)
        self.assertEqual(self.widget.domain_editor.model().data(idx, Qt.DisplayRole), temp)
        self.widget.domain_editor.model().setData(idx, '', Qt.EditRole)
        self.assertEqual(self.widget.domain_editor.model().data(idx, Qt.DisplayRole), temp)

    def test_invalid_role_mode(self):
        if False:
            return 10
        self.open_dataset('iris')
        model = self.widget.domain_editor.model()
        idx = model.createIndex(1, 0)
        self.assertFalse(model.setData(idx, Qt.StatusTipRole, ''))
        self.assertIsNone(model.data(idx, Qt.StatusTipRole))

    def test_context_match_includes_variable_values(self):
        if False:
            for i in range(10):
                print('nop')
        file1 = 'var\na b\n\na\n'
        file2 = 'var\na b c\n\na\n'
        editor = self.widget.domain_editor
        idx = self.widget.domain_editor.model().createIndex(0, 3)
        with named_file(file1, suffix='.tab') as filename:
            self.open_dataset(filename)
            self.assertEqual(editor.model().data(idx, Qt.DisplayRole), 'a, b')
        with named_file(file2, suffix='.tab') as filename:
            self.open_dataset(filename)
            self.assertEqual(editor.model().data(idx, Qt.DisplayRole), 'a, b, c')

    def test_check_datetime_disabled(self):
        if False:
            while True:
                i = 10
        '\n        Datetime option is disable if numerical is disabled as well.\n        GH-2050 (code fixes)\n        GH-2120\n        '
        dat = '            01.08.16\t42.15\tneumann\t2017-02-20\n            03.08.16\t16.08\tneumann\t2017-02-21\n            04.08.16\t23.04\tneumann\t2017-02-22\n            03.09.16\t48.84\tturing\t2017-02-23\n            02.02.17\t23.16\tturing\t2017-02-24'
        with named_file(dat, suffix='.tab') as filename:
            self.open_dataset(filename)
            domain_editor = self.widget.domain_editor
            idx = lambda x: self.widget.domain_editor.model().createIndex(x, 1)
            qcombobox = QComboBox()
            combo = ComboDelegate(domain_editor, VarTableModel.typenames).createEditor(qcombobox, None, idx(2))
            vartype_delegate = VarTypeDelegate(domain_editor, VarTableModel.typenames)
            vartype_delegate.setEditorData(combo, idx(2))
            counts = [4, 2, 4, 2]
            for i in range(4):
                vartype_delegate.setEditorData(combo, idx(i))
                self.assertEqual(combo.count(), counts[i])

    def test_reader_custom_tab(self):
        if False:
            for i in range(10):
                print('nop')
        with named_file('', suffix='.tab') as fn:
            qname = MyCustomTabReader.qualified_name()
            reader = RecentPath(fn, None, None, file_format=qname)
            self.widget = self.create_widget(OWFile, stored_settings={'recent_paths': [reader]})
            self.widget.load_data()
        self.assertFalse(self.widget.Error.missing_reader.is_shown())
        outdata = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(outdata), 150)

    def test_no_reader_extension(self):
        if False:
            for i in range(10):
                print('nop')
        with named_file('', suffix='.xyz_unknown') as fn:
            no_reader = RecentPath(fn, None, None)
            self.widget = self.create_widget(OWFile, stored_settings={'recent_paths': [no_reader]})
            self.widget.load_data()
        self.assertTrue(self.widget.Error.missing_reader.is_shown())

    def test_fail_sheets(self):
        if False:
            return 10
        with named_file('', suffix='.failed_sheet') as fn:
            self.open_dataset(fn)
        self.assertTrue(self.widget.Error.sheet_error.is_shown())

    def test_with_warnings(self):
        if False:
            i = 10
            return i + 15
        with named_file('', suffix='.with_warning') as fn:
            self.open_dataset(fn)
        self.assertTrue(self.widget.Warning.load_warning.is_shown())

    def test_fail(self):
        if False:
            i = 10
            return i + 15
        with named_file('name\nc\n\nstring', suffix='.tab') as fn, patch('Orange.widgets.data.owfile.log.exception') as log:
            self.open_dataset(fn)
            log.assert_called()
        self.assertTrue(self.widget.Error.unknown.is_shown())

    def test_read_format(self):
        if False:
            print('Hello World!')
        iris = Table('iris')

        def open_iris_with_no_spec_format(_a, _b, _c, filters, _e):
            if False:
                i = 10
                return i + 15
            return (iris.__file__, filters.split(';;')[0])
        with patch('AnyQt.QtWidgets.QFileDialog.getOpenFileName', open_iris_with_no_spec_format):
            self.widget.browse_file()
        self.assertIsNone(self.widget.recent_paths[0].file_format)
        self.assertEqual(self.widget.reader_combo.currentText(), DEFAULT_READER_TEXT)

        def open_iris_with_tab(*_):
            if False:
                while True:
                    i = 10
            return (iris.__file__, format_filter(TabReader))
        with patch('AnyQt.QtWidgets.QFileDialog.getOpenFileName', open_iris_with_tab):
            self.widget.browse_file()
        self.assertEqual(self.widget.recent_paths[0].file_format, 'Orange.data.io.TabReader')
        self.assertTrue(self.widget.reader_combo.currentText().startswith('Tab-separated'))

    def test_no_specified_reader(self):
        if False:
            return 10
        with named_file('', suffix='.tab') as fn:
            no_class = RecentPath(fn, None, None, file_format='not.a.file.reader.class')
            self.widget = self.create_widget(OWFile, stored_settings={'recent_paths': [no_class]})
            self.widget.load_data()
        self.assertTrue(self.widget.Error.missing_reader.is_shown())
        self.assertEqual(self.widget.reader_combo.currentText(), 'not.a.file.reader.class')

    def test_select_reader(self):
        if False:
            i = 10
            return i + 15
        filename = FileFormat.locate('iris.tab', dataset_dirs)
        no_class = RecentPath(filename, None, None, file_format='not.a.file.reader.class')
        self.widget = self.create_widget(OWFile, stored_settings={'recent_paths': [no_class]})
        self.widget.load_data()
        len_with_qname = len(self.widget.reader_combo)
        self.assertEqual(self.widget.reader_combo.currentText(), 'not.a.file.reader.class')
        self.assertEqual(self.widget.reader, None)
        self.widget.reader_combo.activated.emit(len_with_qname - 1)
        self.assertEqual(len(self.widget.reader_combo), len_with_qname)
        self.assertEqual(self.widget.reader_combo.currentText(), 'not.a.file.reader.class')
        self.assertEqual(self.widget.reader, None)
        for i in range(len_with_qname):
            text = self.widget.reader_combo.itemText(i)
            if text.startswith('Tab-separated'):
                break
        self.widget.reader_combo.activated.emit(i)
        self.assertEqual(len(self.widget.reader_combo), len_with_qname - 1)
        self.assertTrue(self.widget.reader_combo.currentText().startswith('Tab-separated'))
        self.assertIsInstance(self.widget.reader, TabReader)
        self.widget.reader_combo.activated.emit(0)
        self.assertEqual(len(self.widget.reader_combo), len_with_qname - 1)
        self.assertEqual(self.widget.reader_combo.currentText(), DEFAULT_READER_TEXT)
        self.assertIsInstance(self.widget.reader, TabReader)

    def test_select_reader_errors(self):
        if False:
            print('Hello World!')
        filename = FileFormat.locate('iris.tab', dataset_dirs)
        no_class = RecentPath(filename, None, None, file_format='Orange.data.io.ExcelReader')
        self.widget = self.create_widget(OWFile, stored_settings={'recent_paths': [no_class]})
        self.widget.load_data()
        self.assertIn('Excel', self.widget.reader_combo.currentText())
        self.assertTrue(self.widget.Error.unknown.is_shown())
        self.assertFalse(self.widget.Error.missing_reader.is_shown())

    def test_domain_edit_no_changes(self):
        if False:
            return 10
        self.open_dataset('iris')
        data = self.get_output(self.widget.Outputs.data)
        self.assertTrue(data is self.widget.data)

    def test_domain_edit_on_sparse_data(self):
        if False:
            for i in range(10):
                print('nop')
        iris = Table('iris').to_sparse()
        with named_file('', suffix='.pickle') as fn:
            with open(fn, 'wb') as f:
                pickle.dump(iris, f)
            self.widget.add_path(fn)
            self.widget.load_data()
        output = self.get_output(self.widget.Outputs.data)
        self.assertIsInstance(output, Table)
        self.assertEqual(iris.X.shape, output.X.shape)
        self.assertTrue(sp.issparse(output.X))

    def test_drop_data_when_everything_skipped(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        No data when everything is skipped. Otherwise Select Rows crashes.\n        GH-2237\n        '
        self.open_dataset('iris')
        data = self.get_output(self.widget.Outputs.data)
        self.assertTrue(len(data), 150)
        self.assertTrue(len(data.domain.variables), 5)
        for i in range(5):
            idx = self.widget.domain_editor.model().createIndex(i, 2)
            self.widget.domain_editor.model().setData(idx, 'skip', Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIsNone(data)

    def test_call_deprecated_dialog_formats(self):
        if False:
            print('Hello World!')
        with self.assertWarns(OrangeDeprecationWarning):
            self.assertIn('Tab', dialog_formats())

    def test_add_new_format(self):
        if False:
            return 10
        called = False
        with named_file('', suffix='.tab') as filename:

            def test_format(_sd, _sf, ff, **_):
                if False:
                    print('Hello World!')
                nonlocal called
                called = True
                self.assertIn(FailedSheetsFormat, ff)
                return (filename, TabReader, '')
            with patch('Orange.widgets.data.owfile.open_filename_dialog', test_format):
                self.widget.browse_file()
        self.assertTrue(called)

    def test_domain_editor_conversions(self):
        if False:
            print('Hello World!')
        dat = 'V0\tV1\tV2\tV3\tV4\tV5\tV6\n                 c\tc\td\td\tc\td\td\n                  \t \t \t \t \t \t\n                 3.0\t1.0\t4\ta\t0.0\tx\t1.0\n                 1.0\t2.0\t4\tb\t0.0\ty\t2.0\n                 2.0\t1.0\t7\ta\t0.0\ty\t2.0\n                 0.0\t2.0\t7\ta\t0.0\tz\t2.0'
        with named_file(dat, suffix='.tab') as filename:
            self.open_dataset(filename)
            data1 = self.get_output(self.widget.Outputs.data)
            model = self.widget.domain_editor.model()
            for (i, a) in enumerate(data1.domain.attributes):
                self.assertEqual(str(a), model.data(model.createIndex(i, 0), Qt.DisplayRole))
            model.setData(model.createIndex(0, 1), 'categorical', Qt.EditRole)
            model.setData(model.createIndex(1, 1), 'text', Qt.EditRole)
            model.setData(model.createIndex(2, 1), 'numeric', Qt.EditRole)
            model.setData(model.createIndex(3, 1), 'numeric', Qt.EditRole)
            model.setData(model.createIndex(6, 1), 'numeric', Qt.EditRole)
            self.widget.apply_button.click()
            data2 = self.get_output(self.widget.Outputs.data)
            self.assertEqual(len(data2.domain.attributes[0].values[0]), 1)
            self.assertEqual(len(data2[0].metas[0]), 1)
            self.assertAlmostEqual(float(data1[0][2].value), data2[0][1])
            self.assertAlmostEqual(float(data1[0][6].value), data2[0][5])

    def test_domaineditor_continuous_to_string(self):
        if False:
            print('Hello World!')
        dat = 'V0\nc\n\n1.0\nnan\n3.0'
        with named_file(dat, suffix='.tab') as filename:
            self.open_dataset(filename)
            model = self.widget.domain_editor.model()
            model.setData(model.createIndex(0, 1), 'text', Qt.EditRole)
            self.widget.apply_button.click()
            data = self.get_output(self.widget.Outputs.data)
            self.assertSequenceEqual(data.metas.ravel().tolist(), ['1', '', '3'])

    def test_domaineditor_makes_variables(self):
        if False:
            print('Hello World!')
        dat = 'V0\tV1\nc\td\n\n1.0\t2'
        v0 = StringVariable.make('V0')
        v1 = ContinuousVariable.make('V1')
        with named_file(dat, suffix='.tab') as filename:
            self.open_dataset(filename)
            model = self.widget.domain_editor.model()
            model.setData(model.createIndex(0, 1), 'text', Qt.EditRole)
            model.setData(model.createIndex(1, 1), 'numeric', Qt.EditRole)
            self.widget.apply_button.click()
            data = self.get_output(self.widget.Outputs.data)
            self.assertEqual(data.domain['V0'], v0)
            self.assertEqual(data.domain['V1'], v1)

    def test_url_no_scheme(self):
        if False:
            while True:
                i = 10
        mock_urlreader = Mock(side_effect=ValueError())
        url = 'foo.bar/xxx.csv'
        with patch('Orange.widgets.data.owfile.UrlReader', mock_urlreader):
            self.widget.url_combo.insertItem(0, url)
            self.widget.url_combo.activated.emit(0)
        mock_urlreader.assert_called_once_with('http://' + url)

    def test_adds_origin(self):
        if False:
            return 10
        self.open_dataset('origin1/images')
        data1 = self.get_output(self.widget.Outputs.data)
        attrs = data1.domain['image'].attributes
        self.assertIn('origin', attrs)
        self.assertIn('origin1', attrs['origin'])
        self.open_dataset('origin2/images')
        data2 = self.get_output(self.widget.Outputs.data)
        attrs = data2.domain['image'].attributes
        self.assertIn('origin', attrs)
        self.assertIn('origin2', attrs['origin'])
        attrs = data1.domain['image'].attributes
        self.assertIn('origin', attrs)
        self.assertIn('origin1', attrs['origin'])

    @patch('Orange.widgets.widget.OWWidget.workflowEnv', Mock(return_value={'basedir': getcwd()}))
    def test_open_moved_workflow(self):
        if False:
            while True:
                i = 10
        'Test opening workflow that has been moved to another location\n        (i.e. sent by email), considering data file is stored in the same\n        directory as the workflow.\n        '
        with tempfile.NamedTemporaryFile(dir=getcwd(), delete=False) as temp_file:
            file_name = temp_file.name
        base_name = path.basename(file_name)
        try:
            recent_path = RecentPath(path.join('temp/datasets', base_name), '', path.join('datasets', base_name))
            stored_settings = {'recent_paths': [recent_path]}
            w = self.create_widget(OWFile, stored_settings=stored_settings)
            w.load_data()
            self.assertEqual(w.file_combo.count(), 1)
            self.assertFalse(w.Error.file_not_found.is_shown())
        finally:
            remove(file_name)

    @patch('Orange.widgets.widget.OWWidget.workflowEnv', Mock(return_value={'basedir': getcwd()}))
    def test_files_relocated(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This test testes if paths are relocated correctly\n        '
        with tempfile.NamedTemporaryFile(dir=getcwd(), delete=False) as temp_file:
            file_name = temp_file.name
        base_name = path.basename(file_name)
        try:
            recent_path = RecentPath(path.join('temp/datasets', base_name), '', path.join('datasets', base_name))
            stored_settings = {'recent_paths': [recent_path]}
            w = self.create_widget(OWFile, stored_settings=stored_settings)
            w.load_data()
            self.assertEqual(w.recent_paths[0].relpath, base_name)
            w.workflowEnvChanged('basedir', base_name, base_name)
            self.assertEqual(w.recent_paths[0].relpath, base_name)
        finally:
            remove(file_name)

    def test_sheets(self):
        if False:
            return 10
        widget = self.widget
        combo = widget.sheet_combo
        widget.last_path = lambda : path.join(path.dirname(__file__), '..', '..', '..', 'tests', 'xlsx_files', 'header_0_sheet.xlsx')
        widget._try_load()
        widget.reader.sheet = 'my_sheet'
        widget._select_active_sheet()
        self.assertEqual(combo.itemText(0), 'Sheet1')
        self.assertEqual(combo.itemText(1), 'my_sheet')
        self.assertEqual(combo.itemText(2), 'Sheet3')
        self.assertEqual(combo.currentIndex(), 1)
        widget.reader.sheet = 'no such sheet'
        widget._select_active_sheet()
        self.assertEqual(combo.currentIndex(), 0)

    @patch('os.path.exists', new=lambda _: True)
    def test_warning_from_another_thread(self):
        if False:
            return 10

        def read():
            if False:
                while True:
                    i = 10
            thread = Thread(target=lambda : warnings.warn('warning from another thread'))
            thread.start()
            thread.join()
            return Table(TITANIC_PATH)
        reader = Mock()
        reader.read = read
        self.widget._get_reader = lambda : reader
        self.widget.last_path = lambda : 'foo'
        self.widget._update_sheet_combo = Mock()
        with self.assertWarns(UserWarning):
            self.widget._try_load()
            self.assertFalse(self.widget.Warning.load_warning.is_shown())

    @patch('os.path.exists', new=lambda _: True)
    def test_warning_from_this_thread(self):
        if False:
            for i in range(10):
                print('nop')
        WARNING_MSG = 'warning from this thread'

        def read():
            if False:
                i = 10
                return i + 15
            warnings.warn(WARNING_MSG)
            return Table(TITANIC_PATH)
        reader = Mock()
        reader.read = read
        self.widget._get_reader = lambda : reader
        self.widget.last_path = lambda : 'foo'
        self.widget._update_sheet_combo = Mock()
        self.widget._try_load()
        self.assertTrue(self.widget.Warning.load_warning.is_shown())
        self.assertIn(WARNING_MSG, str(self.widget.Warning.load_warning))

    def test_recent_url_serialization(self):
        if False:
            i = 10
            return i + 15
        with patch.object(self.widget, 'load_data', lambda : None):
            self.widget.url_combo.insertItem(0, 'https://example.com/test.tab')
            self.widget.url_combo.insertItem(1, 'https://example.com/test1.tab')
            self.widget.source = OWFile.URL
            s = self.widget.settingsHandler.pack_data(self.widget)
            self.assertEqual(s['recent_urls'], ['https://example.com/test.tab', 'https://example.com/test1.tab'])
            self.widget.url_combo.lineEdit().clear()
            QTest.keyClicks(self.widget.url_combo, 'https://example.com/test1.tab')
            QTest.keyClick(self.widget.url_combo, Qt.Key_Enter)
            s = self.widget.settingsHandler.pack_data(self.widget)
            self.assertEqual(s['recent_urls'], ['https://example.com/test1.tab', 'https://example.com/test.tab'])

class TestOWFileDropHandler(unittest.TestCase):

    def test_canDropUrl(self):
        if False:
            return 10
        handler = OWFileDropHandler()
        self.assertTrue(handler.canDropUrl(QUrl('https://example.com/test.tab')))
        self.assertTrue(handler.canDropUrl(QUrl.fromLocalFile('test.tab')))

    def test_parametersFromUrl(self):
        if False:
            for i in range(10):
                print('nop')
        handler = OWFileDropHandler()
        r = handler.parametersFromUrl(QUrl('https://example.com/test.tab'))
        self.assertEqual(r['source'], OWFile.URL)
        self.assertEqual(r['recent_urls'], ['https://example.com/test.tab'])
        r = handler.parametersFromUrl(QUrl.fromLocalFile('test.tab'))
        self.assertEqual(r['source'], OWFile.LOCAL_FILE)
        self.assertEqual(r['recent_paths'][0].basename, 'test.tab')
        defs = {'source': OWFile.LOCAL_FILE, 'recent_paths': [RecentPath('/foo.tab', None, None, 'foo.tab'), RecentPath(path.abspath('test.tab'), None, None, 'test.tab')]}
        with patch.object(OWFile.settingsHandler, 'defaults', defs):
            r = handler.parametersFromUrl(QUrl.fromLocalFile('test.tab'))
        self.assertEqual(len(r['recent_paths']), 2)
        self.assertEqual(r['recent_paths'][0].basename, 'test.tab')
if __name__ == '__main__':
    unittest.main()