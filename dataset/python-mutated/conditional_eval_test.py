"""Tests for tfgan.examples.mnist.conditional_eval."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
import conditional_eval

class ConditionalEvalTest(absltest.TestCase):

    def test_build_graph(self):
        if False:
            while True:
                i = 10
        conditional_eval.main(None, run_eval_loop=False)
if __name__ == '__main__':
    absltest.main()