import math
from typing import List, Dict, Any
import numpy
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.nn import util

@SaliencyInterpreter.register('integrated-gradient')
class IntegratedGradient(SaliencyInterpreter):
    """
    Interprets the prediction using Integrated Gradients (https://arxiv.org/abs/1703.01365)

    Registered as a `SaliencyInterpreter` with name "integrated-gradient".
    """

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        if False:
            return 10
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)
        instances_with_grads = dict()
        for (idx, instance) in enumerate(labeled_instances):
            grads = self._integrate_gradients(instance)
            for (key, grad) in grads.items():
                embedding_grad = numpy.sum(grad[0], axis=1)
                norm = numpy.linalg.norm(embedding_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                grads[key] = normalized_grad
            instances_with_grads['instance_' + str(idx + 1)] = grads
        return sanitize(instances_with_grads)

    def _register_hooks(self, alpha: int, embeddings_list: List, token_offsets: List):
        if False:
            print('Hello World!')
        '\n        Register a forward hook on the embedding layer which scales the embeddings by alpha. Used\n        for one term in the Integrated Gradients sum.\n\n        We store the embedding output into the embeddings_list when alpha is zero.  This is used\n        later to element-wise multiply the input by the averaged gradients.\n        '

        def forward_hook(module, inputs, output):
            if False:
                for i in range(10):
                    print('nop')
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach())
            output.mul_(alpha)

        def get_token_offsets(module, inputs, outputs):
            if False:
                while True:
                    i = 10
            offsets = util.get_token_offsets_from_text_field_inputs(inputs)
            if offsets is not None:
                token_offsets.append(offsets)
        handles = []
        embedding_layer = self.predictor.get_interpretable_layer()
        handles.append(embedding_layer.register_forward_hook(forward_hook))
        text_field_embedder = self.predictor.get_interpretable_text_field_embedder()
        handles.append(text_field_embedder.register_forward_hook(get_token_offsets))
        return handles

    def _integrate_gradients(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        if False:
            print('Hello World!')
        '\n        Returns integrated gradients for the given [`Instance`](../../data/instance.md)\n        '
        ig_grads: Dict[str, Any] = {}
        embeddings_list: List[torch.Tensor] = []
        token_offsets: List[torch.Tensor] = []
        steps = 10
        for alpha in numpy.linspace(0, 1.0, num=steps, endpoint=False):
            handles = []
            handles = self._register_hooks(alpha, embeddings_list, token_offsets)
            try:
                grads = self.predictor.get_gradients([instance])[0]
            finally:
                for handle in handles:
                    handle.remove()
            if ig_grads == {}:
                ig_grads = grads
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key]
        for key in ig_grads.keys():
            ig_grads[key] /= steps
        embeddings_list.reverse()
        token_offsets.reverse()
        embeddings_list = self._aggregate_token_embeddings(embeddings_list, token_offsets)
        for (idx, input_embedding) in enumerate(embeddings_list):
            key = 'grad_input_' + str(idx + 1)
            ig_grads[key] *= input_embedding
        return ig_grads