"""Tests for the widget module."""
from fire import testutils
from examples.widget import widget

class WidgetTest(testutils.BaseTestCase):

    def testWidgetWhack(self):
        if False:
            print('Hello World!')
        toy = widget.Widget()
        self.assertEqual(toy.whack(), 'whack!')
        self.assertEqual(toy.whack(3), 'whack! whack! whack!')

    def testWidgetBang(self):
        if False:
            print('Hello World!')
        toy = widget.Widget()
        self.assertEqual(toy.bang(), 'bang bang!')
        self.assertEqual(toy.bang('boom'), 'boom bang!')
if __name__ == '__main__':
    testutils.main()