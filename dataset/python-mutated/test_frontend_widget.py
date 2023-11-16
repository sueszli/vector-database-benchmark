import unittest
import pytest
from qtpy import QtWidgets
from qtconsole.frontend_widget import FrontendWidget
from qtpy.QtTest import QTest
from . import no_display

@pytest.mark.skipif(no_display, reason="Doesn't work without a display")
class TestFrontendWidget(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        ' Create the application for the test case.\n        '
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        cls._app.setQuitOnLastWindowClosed(False)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        ' Exit the application.\n        '
        QtWidgets.QApplication.quit()

    def test_transform_classic_prompt(self):
        if False:
            print('Hello World!')
        ' Test detecting classic prompts.\n        '
        w = FrontendWidget(kind='rich')
        t = w._highlighter.transform_classic_prompt
        self.assertEqual(t('>>> test'), 'test')
        self.assertEqual(t(' >>> test'), 'test')
        self.assertEqual(t('\t >>> test'), 'test')
        self.assertEqual(t(''), '')
        self.assertEqual(t('test'), 'test')
        self.assertEqual(t('... test'), 'test')
        self.assertEqual(t(' ... test'), 'test')
        self.assertEqual(t('  ... test'), 'test')
        self.assertEqual(t('\t ... test'), 'test')
        self.assertEqual(t('>>>test'), '>>>test')
        self.assertEqual(t('>> test'), '>> test')
        self.assertEqual(t('...test'), '...test')
        self.assertEqual(t('.. test'), '.. test')
        self.assertEqual(t('[remote] >>> test'), 'test')
        self.assertEqual(t('[foo] >>> test'), '[foo] >>> test')

    def test_transform_ipy_prompt(self):
        if False:
            while True:
                i = 10
        ' Test detecting IPython prompts.\n        '
        w = FrontendWidget(kind='rich')
        t = w._highlighter.transform_ipy_prompt
        self.assertEqual(t('In [1]: test'), 'test')
        self.assertEqual(t('In [2]: test'), 'test')
        self.assertEqual(t('In [10]: test'), 'test')
        self.assertEqual(t(' In [1]: test'), 'test')
        self.assertEqual(t('\t In [1]: test'), 'test')
        self.assertEqual(t(''), '')
        self.assertEqual(t('test'), 'test')
        self.assertEqual(t('   ...: test'), 'test')
        self.assertEqual(t('    ...: test'), 'test')
        self.assertEqual(t('     ...: test'), 'test')
        self.assertEqual(t('\t   ...: test'), 'test')
        self.assertEqual(t('In [1]:test'), 'In [1]:test')
        self.assertEqual(t('[1]: test'), '[1]: test')
        self.assertEqual(t('In: test'), 'In: test')
        self.assertEqual(t(': test'), ': test')
        self.assertEqual(t('...: test'), '...: test')
        self.assertEqual(t('[remote] In [1]: test'), 'test')
        self.assertEqual(t('[foo] In [1]: test'), '[foo] In [1]: test')