"""Tests for the DataClass decorator."""
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util

@test_util.run_all_in_graph_and_eager_modes
class DataclassTest(tf.test.TestCase):

    def test_docstring_example(self):
        if False:
            return 10

        @tff.utils.dataclass
        class Coords:
            x: tf.Tensor
            y: tf.Tensor

        @tf.function
        def fn(start_coords: Coords) -> Coords:
            if False:
                i = 10
                return i + 15

            def cond(it, _):
                if False:
                    while True:
                        i = 10
                return it < 10

            def body(it, coords):
                if False:
                    i = 10
                    return i + 15
                return (it + 1, Coords(x=coords.x + 1, y=coords.y + 2))
            return tf.while_loop(cond, body, loop_vars=(0, start_coords))[1]
        start_coords = Coords(x=tf.constant(0), y=tf.constant(0))
        end_coords = fn(start_coords)
        with self.subTest('OutputType'):
            self.assertIsInstance(end_coords, Coords)
        end_coords_eval = self.evaluate(end_coords)
        with self.subTest('FirstValue'):
            self.assertEqual(end_coords_eval.x, 10)
        with self.subTest('SecondValue'):
            self.assertEqual(end_coords_eval.y, 20)

    def test_docstring_preservation(self):
        if False:
            for i in range(10):
                print('nop')

        @tff.utils.dataclass
        class Coords:
            """A coordinate grid."""
            x: tf.Tensor
            y: tf.Tensor
        self.assertEqual(Coords.__doc__, 'A coordinate grid.')
if __name__ == '__main__':
    tf.test.main()