import torch
import numpy as np
from .inverter_util import RelevancePropagator

class EpsilonLrp(object):

    def __init__(self, model):
        if False:
            print('Hello World!')
        self.model = InnvestigateModel(model)

    def explain(self, inp, ind=None, raw_inp=None):
        if False:
            for i in range(10):
                print('nop')
        (predicitions, saliency) = self.model.innvestigate(inp, ind)
        return saliency

class InnvestigateModel(torch.nn.Module):
    """
    ATTENTION:
        Currently, innvestigating a network only works if all
        layers that have to be inverted are specified explicitly
        and registered as a module. If., for example,
        only the functional max_poolnd is used, the inversion will not work.
    """

    def __init__(self, the_model, lrp_exponent=1, beta=0.5, epsilon=1e-06, method='e-rule'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Model wrapper for pytorch models to 'innvestigate' them\n        with layer-wise relevance propagation (LRP) as introduced by Bach et. al\n        (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).\n        Given a class level probability produced by the model under consideration,\n        the LRP algorithm attributes this probability to the nodes in each layer.\n        This allows for visualizing the relevance of input pixels on the resulting\n        class probability.\n\n        Args:\n            the_model: Pytorch model, e.g. a pytorch.nn.Sequential consisting of\n                        different layers. Not all layers are supported yet.\n            lrp_exponent: Exponent for rescaling the importance values per node\n                            in a layer when using the e-rule method.\n            beta: Beta value allows for placing more (large beta) emphasis on\n                    nodes that positively contribute to the activation of a given node\n                    in the subsequent layer. Low beta value allows for placing more emphasis\n                    on inhibitory neurons in a layer. Only relevant for method 'b-rule'.\n            epsilon: Stabilizing term to avoid numerical instabilities if the norm (denominator\n                    for distributing the relevance) is close to zero.\n            method: Different rules for the LRP algorithm, b-rule allows for placing\n                    more or less focus on positive / negative contributions, whereas\n                    the e-rule treats them equally. For more information,\n                    see the paper linked above.\n        "
        super(InnvestigateModel, self).__init__()
        self.model = the_model
        self.device = torch.device('cpu', 0)
        self.prediction = None
        self.r_values_per_layer = None
        self.only_max_score = None
        self.inverter = RelevancePropagator(lrp_exponent=lrp_exponent, beta=beta, method=method, epsilon=epsilon, device=self.device)
        self.register_hooks(self.model)
        if method == 'b-rule' and float(beta) in (-1.0, 0):
            which = 'positive' if beta == -1 else 'negative'
            which_opp = 'negative' if beta == -1 else 'positive'
            print('WARNING: With the chosen beta value, only ' + which + ' contributions will be taken into account.\nHence, if in any layer only ' + which_opp + ' contributions exist, the overall relevance will not be conserved.\n')

    def cuda(self, device=None):
        if False:
            i = 10
            return i + 15
        self.device = torch.device('cuda', device)
        self.inverter.device = self.device
        return super(InnvestigateModel, self).cuda(device)

    def cpu(self):
        if False:
            return 10
        self.device = torch.device('cpu', 0)
        self.inverter.device = self.device
        return super(InnvestigateModel, self).cpu()

    def register_hooks(self, parent_module):
        if False:
            while True:
                i = 10
        '\n        Recursively unrolls a model and registers the required\n        hooks to save all the necessary values for LRP in the forward pass.\n\n        Args:\n            parent_module: Model to unroll and register hooks for.\n\n        Returns:\n            None\n\n        '
        for mod in parent_module.children():
            if list(mod.children()):
                self.register_hooks(mod)
                continue
            mod.register_forward_hook(self.inverter.get_layer_fwd_hook(mod))
            if isinstance(mod, torch.nn.ReLU):
                mod.register_backward_hook(self.relu_hook_function)

    @staticmethod
    def relu_hook_function(module, grad_in, grad_out):
        if False:
            for i in range(10):
                print('nop')
        '\n        If there is a negative gradient, change it to zero.\n        '
        return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, in_tensor):
        if False:
            return 10
        '\n        The innvestigate wrapper returns the same prediction as the\n        original model, but wraps the model call method in the evaluate\n        method to save the last prediction.\n\n        Args:\n            in_tensor: Model input to pass through the pytorch model.\n\n        Returns:\n            Model output.\n        '
        return self.evaluate(in_tensor)

    def evaluate(self, in_tensor):
        if False:
            print('Hello World!')
        '\n        Evaluates the model on a new input. The registered forward hooks will\n        save all the data that is necessary to compute the relevance per neuron per layer.\n\n        Args:\n            in_tensor: New input for which to predict an output.\n\n        Returns:\n            Model prediction\n        '
        self.inverter.reset_module_list()
        self.prediction = self.model(in_tensor)
        return self.prediction

    def get_r_values_per_layer(self):
        if False:
            for i in range(10):
                print('nop')
        if self.r_values_per_layer is None:
            print('No relevances have been calculated yet, returning None in get_r_values_per_layer.')
        return self.r_values_per_layer

    def innvestigate(self, in_tensor=None, rel_for_class=None):
        if False:
            return 10
        "\n        Method for 'innvestigating' the model with the LRP rule chosen at\n        the initialization of the InnvestigateModel.\n        Args:\n            in_tensor: Input for which to evaluate the LRP algorithm.\n                        If input is None, the last evaluation is used.\n                        If no evaluation has been performed since initialization,\n                        an error is raised.\n            rel_for_class (int): Index of the class for which the relevance\n                        distribution is to be analyzed. If None, the 'winning' class\n                        is used for indexing.\n\n        Returns:\n            Model output and relevances of nodes in the input layer.\n            In order to get relevance distributions in other layers, use\n            the get_r_values_per_layer method.\n        "
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None
        with torch.no_grad():
            if in_tensor is None and self.prediction is None:
                raise RuntimeError('Model needs to be evaluated at least once before an innvestigation can be performed. Please evaluate model first or call innvestigate with a new input to evaluate.')
            if in_tensor is not None:
                self.evaluate(in_tensor)
            if rel_for_class is None:
                org_shape = self.prediction.size()
                self.prediction = self.prediction.view(org_shape[0], -1)
                (max_v, _) = torch.max(self.prediction, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                relevance_tensor = only_max_score.view(org_shape)
                self.prediction.view(org_shape)
            else:
                org_shape = self.prediction.size()
                self.prediction = self.prediction.view(org_shape[0], -1)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[:, rel_for_class] += self.prediction[:, rel_for_class]
                relevance_tensor = only_max_score.view(org_shape)
                self.prediction.view(org_shape)
            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor
            r_values_per_layer = [relevance]
            for layer in rev_model:
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                r_values_per_layer.append(relevance.cpu())
            self.r_values_per_layer = r_values_per_layer
            del relevance
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return (self.prediction, r_values_per_layer[-1])

    def forward(self, in_tensor):
        if False:
            return 10
        return self.model.forward(in_tensor)

    def extra_repr(self):
        if False:
            return 10
        'Set the extra representation of the module\n\n        To print customized extra information, you should re-implement\n        this method in your own modules. Both single-line and multi-line\n        strings are acceptable.\n        '
        return self.model.extra_repr()