import math
from typing import Dict, Any
import numpy
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.predictors import Predictor

@SaliencyInterpreter.register('smooth-gradient')
class SmoothGradient(SaliencyInterpreter):
    """
    Interprets the prediction using SmoothGrad (https://arxiv.org/abs/1706.03825)

    Registered as a `SaliencyInterpreter` with name "smooth-gradient".
    """

    def __init__(self, predictor: Predictor) -> None:
        if False:
            print('Hello World!')
        super().__init__(predictor)
        self.stdev = 0.01
        self.num_samples = 10

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        if False:
            return 10
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)
        instances_with_grads = dict()
        for (idx, instance) in enumerate(labeled_instances):
            grads = self._smooth_grads(instance)
            for (key, grad) in grads.items():
                embedding_grad = numpy.sum(grad[0], axis=1)
                norm = numpy.linalg.norm(embedding_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                grads[key] = normalized_grad
            instances_with_grads['instance_' + str(idx + 1)] = grads
        return sanitize(instances_with_grads)

    def _register_forward_hook(self, stdev: float):
        if False:
            print('Hello World!')
        '\n        Register a forward hook on the embedding layer which adds random noise to every embedding.\n        Used for one term in the SmoothGrad sum.\n        '

        def forward_hook(module, inputs, output):
            if False:
                print('Hello World!')
            scale = output.detach().max() - output.detach().min()
            noise = torch.randn(output.shape, device=output.device) * stdev * scale
            output.add_(noise)
        embedding_layer = self.predictor.get_interpretable_layer()
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _smooth_grads(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        if False:
            return 10
        total_gradients: Dict[str, Any] = {}
        for _ in range(self.num_samples):
            handle = self._register_forward_hook(self.stdev)
            try:
                grads = self.predictor.get_gradients([instance])[0]
            finally:
                handle.remove()
            if total_gradients == {}:
                total_gradients = grads
            else:
                for key in grads.keys():
                    total_gradients[key] += grads[key]
        for key in total_gradients.keys():
            total_gradients[key] /= self.num_samples
        return total_gradients