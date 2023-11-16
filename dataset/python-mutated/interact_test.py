"""Tests for the interact module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import interact
from fire import testutils
import mock
try:
    import IPython
    INTERACT_METHOD = 'IPython.start_ipython'
except ImportError:
    INTERACT_METHOD = 'code.InteractiveConsole'

class InteractTest(testutils.BaseTestCase):

    @mock.patch(INTERACT_METHOD)
    def testInteract(self, mock_interact_method):
        if False:
            while True:
                i = 10
        self.assertFalse(mock_interact_method.called)
        interact.Embed({})
        self.assertTrue(mock_interact_method.called)

    @mock.patch(INTERACT_METHOD)
    def testInteractVariables(self, mock_interact_method):
        if False:
            print('Hello World!')
        self.assertFalse(mock_interact_method.called)
        interact.Embed({'count': 10, 'mock': mock})
        self.assertTrue(mock_interact_method.called)
if __name__ == '__main__':
    testutils.main()