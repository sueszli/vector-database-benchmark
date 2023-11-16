"""Basic list operations."""
import tensorflow as tf
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.autograph.tests import reference_test_base

def type_not_annotated(n):
    if False:
        return 10
    l = []
    for i in range(n):
        l.append(i)
    return special_functions.stack(l, strict=False)

def element_access():
    if False:
        while True:
            i = 10
    l = []
    l.append(1)
    l.append(2)
    l.append(3)
    directives.set_element_type(l, tf.int32)
    return 2 * l[1]

def element_update():
    if False:
        print('Hello World!')
    l = []
    l.append(1)
    l.append(2)
    l.append(3)
    directives.set_element_type(l, tf.int32)
    l[1] = 5
    return special_functions.stack(l, strict=False)

def simple_fill(n):
    if False:
        for i in range(10):
            print('nop')
    l = []
    directives.set_element_type(l, tf.int32)
    for i in range(n):
        l.append(i)
    return special_functions.stack(l, strict=False)

def nested_fill(m, n):
    if False:
        i = 10
        return i + 15
    mat = []
    directives.set_element_type(mat, tf.int32)
    for _ in range(m):
        l = []
        directives.set_element_type(l, tf.int32)
        for j in range(n):
            l.append(j)
        mat.append(special_functions.stack(l, strict=False))
    return special_functions.stack(mat, strict=False)

def read_write_loop(n):
    if False:
        while True:
            i = 10
    l = []
    l.append(1)
    l.append(1)
    directives.set_element_type(l, tf.int32)
    for i in range(2, n):
        l.append(l[i - 1] + l[i - 2])
        l[i - 2] = -l[i - 2]
    return special_functions.stack(l, strict=False)

def simple_empty(n):
    if False:
        i = 10
        return i + 15
    l = []
    l.append(1)
    l.append(2)
    l.append(3)
    l.append(4)
    directives.set_element_type(l, tf.int32, ())
    s = 0
    for _ in range(n):
        s += l.pop()
    return (special_functions.stack(l, strict=False), s)

def mutation(t, n):
    if False:
        print('Hello World!')
    for i in range(n):
        t[i] = i
    return t

class ReferenceTest(reference_test_base.TestCase):

    def setUp(self):
        if False:
            return 10
        super(ReferenceTest, self).setUp()
        self.autograph_opts = tf.autograph.experimental.Feature.LISTS

    def test_tensor_mutation(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertConvertedMatchesNative(mutation, [0] * 10, 10)

    def test_basic(self):
        if False:
            print('Hello World!')
        self.all_inputs_tensors = True
        self.assertFunctionMatchesEager(element_access)
        self.assertFunctionMatchesEager(element_update)
        with self.assertRaisesRegex(ValueError, 'cannot stack a list without knowing its element type; use set_element_type to annotate it'):
            self.function(type_not_annotated)(3)
        self.assertFunctionMatchesEager(simple_fill, 5)
        self.assertFunctionMatchesEager(nested_fill, 5, 3)
        self.assertFunctionMatchesEager(read_write_loop, 4)
        self.assertFunctionMatchesEager(simple_empty, 0)
        self.assertFunctionMatchesEager(simple_empty, 2)
        self.assertFunctionMatchesEager(simple_empty, 4)
        with self.assertRaises(ValueError):
            self.function(simple_fill)(0)
        with self.assertRaises(ValueError):
            self.function(nested_fill)(0, 3)
if __name__ == '__main__':
    tf.test.main()