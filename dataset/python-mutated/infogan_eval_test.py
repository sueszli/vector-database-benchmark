"""Tests for tfgan.examples.mnist.infogan_eval."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
import infogan_eval

class MnistInfoGANEvalTest(absltest.TestCase):

    def test_build_graph(self):
        if False:
            for i in range(10):
                print('nop')
        infogan_eval.main(None, run_eval_loop=False)
if __name__ == '__main__':
    absltest.main()