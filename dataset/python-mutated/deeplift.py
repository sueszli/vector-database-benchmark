from .backprop import GradxInputExplainer
import types
import torch.nn.functional as F
from torch.autograd import Variable

class DeepLIFTRescaleExplainer(GradxInputExplainer):

    def __init__(self, model):
        if False:
            return 10
        super(DeepLIFTRescaleExplainer, self).__init__(model)
        self._prepare_reference()
        self.baseline_inp = None
        self._override_backward()

    def _prepare_reference(self):
        if False:
            for i in range(10):
                print('nop')

        def init_refs(m):
            if False:
                print('Hello World!')
            name = m.__class__.__name__
            if name.find('ReLU') != -1:
                m.ref_inp_list = []
                m.ref_out_list = []

        def ref_forward(self, x):
            if False:
                for i in range(10):
                    print('nop')
            self.ref_inp_list.append(x.data.clone())
            out = F.relu(x)
            self.ref_out_list.append(out.data.clone())
            return out

        def ref_replace(m):
            if False:
                return 10
            name = m.__class__.__name__
            if name.find('ReLU') != -1:
                m.forward = types.MethodType(ref_forward, m)
        self.model.apply(init_refs)
        self.model.apply(ref_replace)

    def _reset_preference(self):
        if False:
            i = 10
            return i + 15

        def reset_refs(m):
            if False:
                while True:
                    i = 10
            name = m.__class__.__name__
            if name.find('ReLU') != -1:
                m.ref_inp_list = []
                m.ref_out_list = []
        self.model.apply(reset_refs)

    def _baseline_forward(self, inp):
        if False:
            print('Hello World!')
        if self.baseline_inp is None:
            self.baseline_inp = inp.data.clone()
            self.baseline_inp.fill_(0.0)
            self.baseline_inp = Variable(self.baseline_inp)
        else:
            self.baseline_inp.fill_(0.0)
        _ = self.model(self.baseline_inp)

    def _override_backward(self):
        if False:
            print('Hello World!')

        def new_backward(self, grad_out):
            if False:
                return 10
            (ref_inp, inp) = self.ref_inp_list
            (ref_out, out) = self.ref_out_list
            delta_out = out - ref_out
            delta_in = inp - ref_inp
            g1 = (delta_in.abs() > 1e-05).float() * grad_out * delta_out / delta_in
            mask = (ref_inp + inp > 0).float()
            g2 = (delta_in.abs() <= 1e-05).float() * 0.5 * mask * grad_out
            return g1 + g2

        def backward_replace(m):
            if False:
                print('Hello World!')
            name = m.__class__.__name__
            if name.find('ReLU') != -1:
                m.backward = types.MethodType(new_backward, m)
        self.model.apply(backward_replace)

    def explain(self, inp, ind=None, raw_inp=None):
        if False:
            while True:
                i = 10
        self._reset_preference()
        self._baseline_forward(inp)
        g = super(DeepLIFTRescaleExplainer, self).explain(inp, ind)
        return g