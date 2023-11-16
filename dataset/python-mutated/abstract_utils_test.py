"""Tests for abstract_utils.py."""
from pytype import config
from pytype.abstract import abstract_utils
from pytype.tests import test_base
from pytype.tests import test_utils
import unittest

class GetViewsTest(test_base.UnitTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        options = config.Options.create(python_version=self.python_version)
        self._ctx = test_utils.make_context(options)

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        v1 = self._ctx.program.NewVariable([self._ctx.convert.unsolvable], [], self._ctx.root_node)
        v2 = self._ctx.program.NewVariable([self._ctx.convert.int_type, self._ctx.convert.str_type], [], self._ctx.root_node)
        views = list(abstract_utils.get_views([v1, v2], self._ctx.root_node))
        self.assertCountEqual([{v1: views[0][v1], v2: views[0][v2]}, {v1: views[1][v1], v2: views[1][v2]}], [{v1: v1.bindings[0], v2: v2.bindings[0]}, {v1: v1.bindings[0], v2: v2.bindings[1]}])

    def _test_optimized(self, skip_future_value, expected_num_views):
        if False:
            print('Hello World!')
        v1 = self._ctx.program.NewVariable([self._ctx.convert.unsolvable], [], self._ctx.root_node)
        v2 = self._ctx.program.NewVariable([self._ctx.convert.int_type, self._ctx.convert.str_type], [], self._ctx.root_node)
        views = abstract_utils.get_views([v1, v2], self._ctx.root_node)
        skip_future = None
        view_markers = []
        while True:
            try:
                view = views.send(skip_future)
            except StopIteration:
                break
            view_markers.append(view[v1])
            skip_future = skip_future_value
        self.assertEqual(len(view_markers), expected_num_views)

    def test_skip(self):
        if False:
            print('Hello World!')
        self._test_optimized(skip_future_value=True, expected_num_views=1)

    def test_no_skip(self):
        if False:
            while True:
                i = 10
        self._test_optimized(skip_future_value=False, expected_num_views=2)
if __name__ == '__main__':
    unittest.main()