"""Tests for the collector module."""
from fire import testutils
from examples.widget import collector
from examples.widget import widget

class CollectorTest(testutils.BaseTestCase):

    def testCollectorHasWidget(self):
        if False:
            while True:
                i = 10
        col = collector.Collector()
        self.assertIsInstance(col.widget, widget.Widget)

    def testCollectorWantsMoreWidgets(self):
        if False:
            while True:
                i = 10
        col = collector.Collector()
        self.assertEqual(col.desired_widget_count, 10)

    def testCollectorGetsWantedWidgets(self):
        if False:
            print('Hello World!')
        col = collector.Collector()
        self.assertEqual(len(col.collect_widgets()), 10)
if __name__ == '__main__':
    testutils.main()