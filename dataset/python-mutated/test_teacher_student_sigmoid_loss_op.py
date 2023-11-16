from math import exp, log
import numpy as np
from op_test import OpTest
from scipy.special import logit

class TestTeacherStudentSigmoidLossOp(OpTest):
    """
    Test teacher_student_sigmoid_loss with discrete one-hot labels.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'teacher_student_sigmoid_loss'
        batch_size = 100
        num_classes = 1
        self.inputs = {'X': logit(np.random.uniform(0, 1, (batch_size, num_classes)).astype('float64')), 'Label': np.random.uniform(0, 2, (batch_size, num_classes)).astype('float64')}
        outs = []
        for (index, label) in enumerate(self.inputs['Label']):
            x = self.inputs['X'][index]
            if label < -1.0:
                outs.append(max(x, 0.0) + log(1.0 + exp(-abs(x))))
            elif label < 0.0:
                outs.append(max(x, 0.0) - x + log(1.0 + exp(-abs(x))))
            elif label < 1.0:
                outs.append(max(x, 0.0) + log(1.0 + exp(-abs(x))) + max(x, 0.0) - x * label + log(1.0 + exp(-abs(x))))
            else:
                outs.append(max(x, 0.0) - x + log(1.0 + exp(-abs(x))) + max(x, 0.0) - x * (label - 1.0) + log(1.0 + exp(-abs(x))))
        self.outputs = {'Y': np.array(outs)}

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Y', numeric_grad_delta=0.005)