import numpy as np
from keras import backend
from keras import ops
from keras import testing
from keras.optimizers.nadam import Nadam

class NadamTest(testing.TestCase):

    def test_config(self):
        if False:
            i = 10
            return i + 15
        optimizer = Nadam(learning_rate=0.5, beta_1=0.5, beta_2=0.67, epsilon=1e-05)
        self.run_class_serialization_test(optimizer)

    def test_single_step(self):
        if False:
            for i in range(10):
                print('nop')
        optimizer = Nadam(learning_rate=0.5)
        grads = ops.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0])
        optimizer.apply_gradients(zip([grads], [vars]))
        self.assertAllClose(vars, [0.4686, 1.4686, 2.4686, 3.4686], rtol=0.0001, atol=0.0001)

    def test_weight_decay(self):
        if False:
            return 10
        (grads, var1, var2, var3) = (ops.zeros(()), backend.Variable(2.0), backend.Variable(2.0, name='exclude'), backend.Variable(2.0))
        optimizer_1 = Nadam(learning_rate=1.0, weight_decay=0.004)
        optimizer_1.apply_gradients(zip([grads], [var1]))
        optimizer_2 = Nadam(learning_rate=1.0, weight_decay=0.004)
        optimizer_2.exclude_from_weight_decay(var_names=['exclude'])
        optimizer_2.apply_gradients(zip([grads, grads], [var1, var2]))
        optimizer_3 = Nadam(learning_rate=1.0, weight_decay=0.004)
        optimizer_3.exclude_from_weight_decay(var_list=[var3])
        optimizer_3.apply_gradients(zip([grads, grads], [var1, var3]))
        self.assertAlmostEqual(var1.numpy(), 1.9760959, decimal=6)
        self.assertAlmostEqual(var2.numpy(), 2.0, decimal=6)
        self.assertAlmostEqual(var3.numpy(), 2.0, decimal=6)

    def test_correctness_with_golden(self):
        if False:
            i = 10
            return i + 15
        optimizer = Nadam(learning_rate=0.5, beta_1=0.5, beta_2=0.67, epsilon=1e-05)
        x = backend.Variable(np.ones([10]))
        grads = ops.arange(0.1, 1.1, 0.1)
        first_grads = ops.full((10,), 0.01)
        golden = np.array([[0.4281, 0.4281, 0.4281, 0.4281, 0.4281, 0.4281, 0.4281, 0.4281, 0.4281, 0.4281], [-0.1738, -0.1731, -0.1726, -0.1723, -0.1721, -0.172, -0.1719, -0.1718, -0.1718, -0.1717], [-0.7115, -0.7103, -0.7096, -0.7092, -0.709, -0.7088, -0.7086, -0.7085, -0.7085, -0.7084], [-1.2335, -1.2322, -1.2313, -1.2309, -1.2306, -1.2304, -1.2302, -1.2301, -1.23, -1.2299], [-1.7492, -1.7478, -1.7469, -1.7464, -1.7461, -1.7459, -1.7457, -1.7456, -1.7455, -1.7454]])
        optimizer.apply_gradients(zip([first_grads], [x]))
        for i in range(5):
            self.assertAllClose(x, golden[i], rtol=0.0005, atol=0.0005)
            optimizer.apply_gradients(zip([grads], [x]))

    def test_clip_norm(self):
        if False:
            i = 10
            return i + 15
        optimizer = Nadam(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2 ** 0.5 / 2, 2 ** 0.5 / 2])

    def test_clip_value(self):
        if False:
            while True:
                i = 10
        optimizer = Nadam(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])