import ivy
from ivy.functional.frontends.xgboost.objective.regression_loss import LogisticRegression
from ivy.functional.frontends.xgboost.linear.updater_coordinate import coordinate_updater
from copy import deepcopy

class GBLinear:

    def __init__(self, params=None, compile=False, cache=None):
        if False:
            return 10
        self.num_boosted_rounds = 0
        self.updater = coordinate_updater
        self.obj = LogisticRegression()
        self.base_score = self.obj.prob_to_margin(params['base_score'])
        self.num_inst = params['num_instances']
        self.sum_instance_weight_ = self.num_inst
        self.scale_pos_weight = 1.0 if not params['scale_pos_weight'] else params['scale_pos_weight']
        self.is_null_weights = True
        self.is_converged_ = False
        self.tolerance = 0.0
        self.num_output_group = params['num_output_group']
        self.num_feature = params['num_feature']
        self.weight = ivy.zeros((self.num_feature + 1, self.num_output_group), dtype=ivy.float32)
        self.prev_weight = deepcopy(self.weight)
        self.base_margin = params['base_margin'] if params['base_margin'] else self.base_score
        self.learning_rate = params['learning_rate']
        self.reg_lambda_denorm = self.sum_instance_weight_ * params['reg_lambda']
        self.reg_alpha_denorm = self.sum_instance_weight_ * params['reg_alpha']
        self.compile = compile
        if self.compile:
            backend_compile = True if ivy.current_backend_str() != 'torch' else False
            self._comp_pred = ivy.trace_graph(_pred, backend_compile=backend_compile)
            self._comp_get_gradient = ivy.trace_graph(_get_gradient, backend_compile=backend_compile, static_argnums=(0,))
            self._comp_updater = ivy.trace_graph(self.updater, backend_compile=backend_compile)
            pred = self._comp_pred(cache[0], self.weight, self.base_margin)
            gpair = self._comp_get_gradient(self.obj, pred, cache[1], self.scale_pos_weight)
            self._comp_updater(gpair, cache[0], self.learning_rate, self.weight, self.num_feature, 0, self.reg_alpha_denorm, self.reg_lambda_denorm)

    def boosted_rounds(self):
        if False:
            for i in range(10):
                print('nop')
        return self.num_boosted_rounds

    def model_fitted(self):
        if False:
            for i in range(10):
                print('nop')
        return self.num_boosted_rounds != 0

    def check_convergence(self):
        if False:
            for i in range(10):
                print('nop')
        if self.tolerance == 0.0:
            return False
        elif self.is_converged_:
            return True
        largest_dw = ivy.max(ivy.abs(self.weight - self.prev_weight))
        self.prev_weight = self.weight.copy()
        self.is_converged_ = largest_dw <= self.tolerance
        return self.is_converged_

    def pred(self, data):
        if False:
            while True:
                i = 10
        args = (data, self.weight, self.base_margin)
        if self.compile:
            return self._comp_pred(*args)
        else:
            return _pred(*args)

    def get_gradient(self, pred, label):
        if False:
            i = 10
            return i + 15
        args = (self.obj, pred, label, self.scale_pos_weight)
        if self.compile:
            return self._comp_get_gradient(*args)
        else:
            return _get_gradient(*args)

    def do_boost(self, data, gpair, iter):
        if False:
            print('Hello World!')
        if not self.check_convergence():
            self.num_boosted_rounds += 1
            args = (gpair, data, self.learning_rate, self.weight, self.num_feature, iter, self.reg_alpha_denorm, self.reg_lambda_denorm)
            if self.compile:
                self.weight = self._comp_updater(*args)
            else:
                self.weight = self.updater(*args)

def _get_gradient(obj, pred, label, scale_pos_weight):
    if False:
        while True:
            i = 10
    p = obj.pred_transform(pred)
    w = 1.0
    w_scaled = ivy.where(label == 1.0, w * scale_pos_weight, w)
    return ivy.hstack([obj.first_order_gradient(p, label) * w_scaled, obj.second_order_gradient(p, label) * w_scaled])

def _pred(dt, w, base):
    if False:
        i = 10
        return i + 15
    return ivy.matmul(dt, w[:-1]) + w[-1] + base