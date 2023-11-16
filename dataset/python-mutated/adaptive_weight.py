import numpy as np
from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer
from caffe2.python.regularizer import BoundedGradientProjection, LogBarrier
'\nImplementation of adaptive weighting: https://arxiv.org/pdf/1705.07115.pdf\n'

class AdaptiveWeight(ModelLayer):

    def __init__(self, model, input_record, name='adaptive_weight', optimizer=None, weights=None, enable_diagnose=False, estimation_method='log_std', pos_optim_method='log_barrier', reg_lambda=0.1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(model, name, input_record, **kwargs)
        self.output_schema = schema.Scalar(np.float32, self.get_next_blob_reference('adaptive_weight'))
        self.data = self.input_record.field_blobs()
        self.num = len(self.data)
        self.optimizer = optimizer
        if weights is not None:
            assert len(weights) == self.num
        else:
            weights = [1.0 / self.num for _ in range(self.num)]
        assert min(weights) > 0, 'initial weights must be positive'
        self.weights = np.array(weights).astype(np.float32)
        self.estimation_method = str(estimation_method).lower()
        self.pos_optim_method = str(pos_optim_method).lower()
        self.reg_lambda = float(reg_lambda)
        self.enable_diagnose = enable_diagnose
        self.init_func = getattr(self, self.estimation_method + '_init')
        self.weight_func = getattr(self, self.estimation_method + '_weight')
        self.reg_func = getattr(self, self.estimation_method + '_reg')
        self.init_func()
        if self.enable_diagnose:
            self.weight_i = [self.get_next_blob_reference('adaptive_weight_%d' % i) for i in range(self.num)]
            for i in range(self.num):
                self.model.add_ad_hoc_plot_blob(self.weight_i[i])

    def concat_data(self, net):
        if False:
            i = 10
            return i + 15
        reshaped = [net.NextScopedBlob('reshaped_data_%d' % i) for i in range(self.num)]
        for i in range(self.num):
            net.Reshape([self.data[i]], [reshaped[i], net.NextScopedBlob('new_shape_%d' % i)], shape=[1])
        concated = net.NextScopedBlob('concated_data')
        net.Concat(reshaped, [concated, net.NextScopedBlob('concated_new_shape')], axis=0)
        return concated

    def log_std_init(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        mu = 2 log sigma, sigma = standard variance\n        per task objective:\n        min 1 / 2 / e^mu X + mu / 2\n        '
        values = np.log(1.0 / 2.0 / self.weights)
        initializer = ('GivenTensorFill', {'values': values, 'dtype': core.DataType.FLOAT})
        self.mu = self.create_param(param_name='mu', shape=[self.num], initializer=initializer, optimizer=self.optimizer)

    def log_std_weight(self, x, net, weight):
        if False:
            print('Hello World!')
        '\n        min 1 / 2 / e^mu X + mu / 2\n        '
        mu_neg = net.NextScopedBlob('mu_neg')
        net.Negative(self.mu, mu_neg)
        mu_neg_exp = net.NextScopedBlob('mu_neg_exp')
        net.Exp(mu_neg, mu_neg_exp)
        net.Scale(mu_neg_exp, weight, scale=0.5)

    def log_std_reg(self, net, reg):
        if False:
            i = 10
            return i + 15
        net.Scale(self.mu, reg, scale=0.5)

    def inv_var_init(self):
        if False:
            return 10
        '\n        k = 1 / variance\n        per task objective:\n        min 1 / 2 * k  X - 1 / 2 * log k\n        '
        values = 2.0 * self.weights
        initializer = ('GivenTensorFill', {'values': values, 'dtype': core.DataType.FLOAT})
        if self.pos_optim_method == 'log_barrier':
            regularizer = LogBarrier(reg_lambda=self.reg_lambda)
        elif self.pos_optim_method == 'pos_grad_proj':
            regularizer = BoundedGradientProjection(lb=0, left_open=True)
        else:
            raise TypeError('unknown positivity optimization method: {}'.format(self.pos_optim_method))
        self.k = self.create_param(param_name='k', shape=[self.num], initializer=initializer, optimizer=self.optimizer, regularizer=regularizer)

    def inv_var_weight(self, x, net, weight):
        if False:
            return 10
        net.Scale(self.k, weight, scale=0.5)

    def inv_var_reg(self, net, reg):
        if False:
            while True:
                i = 10
        log_k = net.NextScopedBlob('log_k')
        net.Log(self.k, log_k)
        net.Scale(log_k, reg, scale=-0.5)

    def _add_ops_impl(self, net, enable_diagnose):
        if False:
            return 10
        x = self.concat_data(net)
        weight = net.NextScopedBlob('weight')
        reg = net.NextScopedBlob('reg')
        weighted_x = net.NextScopedBlob('weighted_x')
        weighted_x_add_reg = net.NextScopedBlob('weighted_x_add_reg')
        self.weight_func(x, net, weight)
        self.reg_func(net, reg)
        net.Mul([weight, x], weighted_x)
        net.Add([weighted_x, reg], weighted_x_add_reg)
        net.SumElements(weighted_x_add_reg, self.output_schema())
        if enable_diagnose:
            for i in range(self.num):
                net.Slice(weight, self.weight_i[i], starts=[i], ends=[i + 1])

    def add_ops(self, net):
        if False:
            return 10
        self._add_ops_impl(net, self.enable_diagnose)