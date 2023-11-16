import unittest
from Orange.data import Table
from Orange.widgets.data.owpurgedomain import OWPurgeDomain
from Orange.widgets.tests.base import WidgetTest

class TestOWPurgeDomain(WidgetTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.widget = self.create_widget(OWPurgeDomain)
        self.iris = Table('iris')

    def test_minimum_size(self):
        if False:
            for i in range(10):
                print('nop')
        pass
if __name__ == '__main__':
    unittest.main()