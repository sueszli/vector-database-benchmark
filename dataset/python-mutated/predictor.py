from typing import List, Iterator, Dict, Tuple, Any, Type, Union, Optional
import logging
from os import PathLike
import json
import re
from contextlib import contextmanager
import numpy
import torch
from torch.utils.hooks import RemovableHandle
from torch import Tensor
from torch import backends
from allennlp.common import Registrable, plugins
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.batch import Batch
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.nn import util
logger = logging.getLogger(__name__)

class Predictor(Registrable):
    """
    a `Predictor` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool=True) -> None:
        if False:
            print('Hello World!')
        if frozen:
            model.eval()
        self._model = model
        self._dataset_reader = dataset_reader
        self.cuda_device = next(self._model.named_parameters())[1].get_device()
        self._token_offsets: List[Tensor] = []

    def load_line(self, line: str) -> JsonDict:
        if False:
            print('Hello World!')
        '\n        If your inputs are not in JSON-lines format (e.g. you have a CSV)\n        you can override this function to parse them correctly.\n        '
        return json.loads(line)

    def dump_line(self, outputs: JsonDict) -> str:
        if False:
            i = 10
            return i + 15
        "\n        If you don't want your outputs in JSON-lines format\n        you can override this function to output them differently.\n        "
        return json.dumps(outputs) + '\n'

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        if False:
            for i in range(10):
                print('nop')
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]:
        if False:
            while True:
                i = 10
        "\n        Converts incoming json to a [`Instance`](../data/instance.md),\n        runs the model on the newly created instance, and adds labels to the\n        `Instance`s given by the model's output.\n\n        # Returns\n\n        `List[instance]`\n            A list of `Instance`'s.\n        "
        instance = self._json_to_instance(inputs)
        self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.forward_on_instance(instance)
        new_instances = self.predictions_to_labeled_instances(instance, outputs)
        return new_instances

    def get_gradients(self, instances: List[Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if False:
            while True:
                i = 10
        "\n        Gets the gradients of the loss with respect to the model inputs.\n\n        # Parameters\n\n        instances : `List[Instance]`\n\n        # Returns\n\n        `Tuple[Dict[str, Any], Dict[str, Any]]`\n            The first item is a Dict of gradient entries for each input.\n            The keys have the form  `{grad_input_1: ..., grad_input_2: ... }`\n            up to the number of inputs given. The second item is the model's output.\n\n        # Notes\n\n        Takes a `JsonDict` representing the inputs of the model and converts\n        them to [`Instances`](../data/instance.md)), sends these through\n        the model [`forward`](../models/model.md#forward) function after registering hooks on the embedding\n        layer of the model. Calls `backward` on the loss and then removes the\n        hooks.\n        "
        original_param_name_to_requires_grad_dict = {}
        for (param_name, param) in self._model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True
        embedding_gradients: List[Tensor] = []
        hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(embedding_gradients)
        for instance in instances:
            self._dataset_reader.apply_token_indexers(instance)
        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)
        dataset_tensor_dict = util.move_to_device(dataset.as_tensor_dict(), self.cuda_device)
        with backends.cudnn.flags(enabled=False):
            outputs = self._model.make_output_human_readable(self._model.forward(**dataset_tensor_dict))
            loss = outputs['loss']
            for p in self._model.parameters():
                p.grad = None
            loss.backward()
        for hook in hooks:
            hook.remove()
        grad_dict = dict()
        for (idx, grad) in enumerate(embedding_gradients):
            key = 'grad_input_' + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()
        for (param_name, param) in self._model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]
        return (grad_dict, outputs)

    def get_interpretable_layer(self) -> torch.nn.Module:
        if False:
            while True:
                i = 10
        '\n        Returns the input/embedding layer of the model.\n        If the predictor wraps around a non-AllenNLP model,\n        this function should be overridden to specify the correct input/embedding layer.\n        For the cases where the input layer _is_ an embedding layer, this should be the\n        layer 0 of the embedder.\n        '
        try:
            return util.find_embedding_layer(self._model)
        except RuntimeError:
            raise RuntimeError('If the model does not use `TextFieldEmbedder`, please override `get_interpretable_layer` in your predictor to specify the embedding layer.')

    def get_interpretable_text_field_embedder(self) -> torch.nn.Module:
        if False:
            while True:
                i = 10
        '\n        Returns the first `TextFieldEmbedder` of the model.\n        If the predictor wraps around a non-AllenNLP model,\n        this function should be overridden to specify the correct embedder.\n        '
        try:
            return util.find_text_field_embedder(self._model)
        except RuntimeError:
            raise RuntimeError('If the model does not use `TextFieldEmbedder`, please override `get_interpretable_text_field_embedder` in your predictor to specify the embedding layer.')

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        if False:
            i = 10
            return i + 15
        "\n        Registers a backward hook on the embedding layer of the model.  Used to save the gradients\n        of the embeddings for use in get_gradients()\n\n        When there are multiple inputs (e.g., a passage and question), the hook\n        will be called multiple times. We append all the embeddings gradients\n        to a list.\n\n        We additionally add a hook on the _forward_ pass of the model's `TextFieldEmbedder` to save\n        token offsets, if there are any.  Having token offsets means that you're using a mismatched\n        token indexer, so we need to aggregate the gradients across wordpieces in a token.  We do\n        that with a simple sum.\n        "

        def hook_layers(module, grad_in, grad_out):
            if False:
                i = 10
                return i + 15
            grads = grad_out[0]
            if self._token_offsets:
                offsets = self._token_offsets.pop(0)
                (span_grads, span_mask) = util.batched_span_select(grads.contiguous(), offsets)
                span_mask = span_mask.unsqueeze(-1)
                span_grads *= span_mask
                span_grads_sum = span_grads.sum(2)
                span_grads_len = span_mask.sum(2)
                grads = span_grads_sum / torch.clamp_min(span_grads_len, 1)
                grads[(span_grads_len == 0).expand(grads.shape)] = 0
            embedding_gradients.append(grads)

        def get_token_offsets(module, inputs, outputs):
            if False:
                for i in range(10):
                    print('nop')
            offsets = util.get_token_offsets_from_text_field_inputs(inputs)
            if offsets is not None:
                self._token_offsets.append(offsets)
        hooks = []
        text_field_embedder = self.get_interpretable_text_field_embedder()
        hooks.append(text_field_embedder.register_forward_hook(get_token_offsets))
        embedding_layer = self.get_interpretable_layer()
        hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return hooks

    @contextmanager
    def capture_model_internals(self, module_regex: str='.*') -> Iterator[dict]:
        if False:
            while True:
                i = 10
        '\n        Context manager that captures the internal-module outputs of\n        this predictor\'s model. The idea is that you could use it as follows:\n\n        ```\n            with predictor.capture_model_internals() as internals:\n                outputs = predictor.predict_json(inputs)\n\n            return {**outputs, "model_internals": internals}\n        ```\n        '
        results = {}
        hooks = []

        def add_output(idx: int):
            if False:
                return 10

            def _add_output(mod, _, outputs):
                if False:
                    return 10
                results[idx] = {'name': str(mod), 'output': sanitize(outputs)}
            return _add_output
        regex = re.compile(module_regex)
        for (idx, (name, module)) in enumerate(self._model.named_modules()):
            if regex.fullmatch(name) and module != self._model:
                hook = module.register_forward_hook(add_output(idx))
                hooks.append(hook)
        yield results
        for hook in hooks:
            hook.remove()

    def predict_instance(self, instance: Instance) -> JsonDict:
        if False:
            for i in range(10):
                print('nop')
        self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        if False:
            print('Hello World!')
        "\n        This function takes a model's outputs for an Instance, and it labels that instance according\n        to the `outputs`. This function is used to (1) compute gradients of what the model predicted;\n        (2) label the instance for the attack. For example, (a) for the untargeted attack for classification\n        this function labels the instance according to the class with the highest probability; (b) for\n        targeted attack, it directly constructs fields from the given target.\n        The return type is a list because in some tasks there are multiple predictions in the output\n        (e.g., in NER a model predicts multiple spans). In this case, each instance in the returned list of\n        Instances contains an individual entity prediction as the label.\n        "
        raise RuntimeError('implement this method for model interpretations or attacks')

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts a JSON object into an [`Instance`](../data/instance.md)\n        and a `JsonDict` of information which the `Predictor` should pass through,\n        such as tokenized inputs.\n        '
        raise NotImplementedError

    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        if False:
            for i in range(10):
                print('nop')
        instances = self._batch_json_to_instances(inputs)
        return self.predict_batch_instance(instances)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        if False:
            print('Hello World!')
        for instance in instances:
            self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts a list of JSON objects into a list of `Instance`s.\n        By default, this expects that a "batch" consists of a list of JSON blobs which would\n        individually be predicted by `predict_json`. In order to use this method for\n        batch prediction, `_json_to_instance` should be implemented by the subclass, or\n        if the instances have some dependency on each other, this method should be overridden\n        directly.\n        '
        instances = []
        for json_dict in json_dicts:
            instances.append(self._json_to_instance(json_dict))
        return instances

    @classmethod
    def from_path(cls, archive_path: Union[str, PathLike], predictor_name: str=None, cuda_device: int=-1, dataset_reader_to_load: str='validation', frozen: bool=True, import_plugins: bool=True, overrides: Union[str, Dict[str, Any]]='', **kwargs) -> 'Predictor':
        if False:
            i = 10
            return i + 15
        '\n        Instantiate a `Predictor` from an archive path.\n\n        If you need more detailed configuration options, such as overrides,\n        please use `from_archive`.\n\n        # Parameters\n\n        archive_path : `Union[str, PathLike]`\n            The path to the archive.\n        predictor_name : `str`, optional (default=`None`)\n            Name that the predictor is registered as, or None to use the\n            predictor associated with the model.\n        cuda_device : `int`, optional (default=`-1`)\n            If `cuda_device` is >= 0, the model will be loaded onto the\n            corresponding GPU. Otherwise it will be loaded onto the CPU.\n        dataset_reader_to_load : `str`, optional (default=`"validation"`)\n            Which dataset reader to load from the archive, either "train" or\n            "validation".\n        frozen : `bool`, optional (default=`True`)\n            If we should call `model.eval()` when building the predictor.\n        import_plugins : `bool`, optional (default=`True`)\n            If `True`, we attempt to import plugins before loading the predictor.\n            This comes with additional overhead, but means you don\'t need to explicitly\n            import the modules that your predictor depends on as long as those modules\n            can be found by `allennlp.common.plugins.import_plugins()`.\n        overrides : `Union[str, Dict[str, Any]]`, optional (default = `""`)\n            JSON overrides to apply to the unarchived `Params` object.\n        **kwargs : `Any`\n            Additional key-word arguments that will be passed to the `Predictor`\'s\n            `__init__()` method.\n\n        # Returns\n\n        `Predictor`\n            A Predictor instance.\n        '
        if import_plugins:
            plugins.import_plugins()
        return Predictor.from_archive(load_archive(archive_path, cuda_device=cuda_device, overrides=overrides), predictor_name, dataset_reader_to_load=dataset_reader_to_load, frozen=frozen, extra_args=kwargs)

    @classmethod
    def from_archive(cls, archive: Archive, predictor_name: str=None, dataset_reader_to_load: str='validation', frozen: bool=True, extra_args: Optional[Dict[str, Any]]=None) -> 'Predictor':
        if False:
            i = 10
            return i + 15
        '\n        Instantiate a `Predictor` from an [`Archive`](../models/archival.md);\n        that is, from the result of training a model. Optionally specify which `Predictor`\n        subclass; otherwise, we try to find a corresponding predictor in `DEFAULT_PREDICTORS`, or if\n        one is not found, the base class (i.e. `Predictor`) will be used. Optionally specify\n        which [`DatasetReader`](../data/dataset_readers/dataset_reader.md) should be loaded;\n        otherwise, the validation one will be used if it exists followed by the training dataset reader.\n        Optionally specify if the loaded model should be frozen, meaning `model.eval()` will be called.\n        '
        config = archive.config.duplicate()
        if not predictor_name:
            model_type = config.get('model').get('type')
            (model_class, _) = Model.resolve_class_name(model_type)
            predictor_name = model_class.default_predictor
        predictor_class: Type[Predictor] = Predictor.by_name(predictor_name) if predictor_name is not None else cls
        if dataset_reader_to_load == 'validation':
            dataset_reader = archive.validation_dataset_reader
        else:
            dataset_reader = archive.dataset_reader
        model = archive.model
        if frozen:
            model.eval()
        if extra_args is None:
            extra_args = {}
        return predictor_class(model, dataset_reader, **extra_args)