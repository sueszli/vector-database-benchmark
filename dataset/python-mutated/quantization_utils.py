import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.utils import module_to_fqn, fqn_to_module
from typing import Dict, List, Optional
SUPPORTED_MODULES = {nn.Embedding, nn.EmbeddingBag}

def _fetch_all_embeddings(model):
    if False:
        i = 10
        return i + 15
    'Fetches Embedding and EmbeddingBag modules from the model\n    '
    embedding_modules = []
    stack = [model]
    while stack:
        module = stack.pop()
        for (_, child) in module.named_children():
            fqn_name = module_to_fqn(model, child)
            if type(child) in SUPPORTED_MODULES:
                embedding_modules.append((fqn_name, child))
            else:
                stack.append(child)
    return embedding_modules

def post_training_sparse_quantize(model, data_sparsifier_class, sparsify_first=True, select_embeddings: Optional[List[nn.Module]]=None, **sparse_config):
    if False:
        for i in range(10):
            print('nop')
    'Takes in a model and applies sparsification and quantization to only embeddings & embeddingbags.\n    The quantization step can happen before or after sparsification depending on the `sparsify_first` argument.\n\n    Args:\n        - model (nn.Module)\n            model whose embeddings needs to be sparsified\n        - data_sparsifier_class (type of data sparsifier)\n            Type of sparsification that needs to be applied to model\n        - sparsify_first (bool)\n            if true, sparsifies first and then quantizes\n            otherwise, quantizes first and then sparsifies.\n        - select_embeddings (List of Embedding modules)\n            List of embedding modules to in the model to be sparsified & quantized.\n            If None, all embedding modules with be sparsified\n        - sparse_config (Dict)\n            config that will be passed to the constructor of data sparsifier object.\n\n    Note:\n        1. When `sparsify_first=False`, quantization occurs first followed by sparsification.\n            - before sparsifying, the embedding layers are dequantized.\n            - scales and zero-points are saved\n            - embedding layers are sparsified and `squash_mask` is applied\n            - embedding weights are requantized using the saved scales and zero-points\n        2. When `sparsify_first=True`, sparsification occurs first followed by quantization.\n            - embeddings are sparsified first\n            - quantization is applied on the sparsified embeddings\n    '
    data_sparsifier = data_sparsifier_class(**sparse_config)
    if select_embeddings is None:
        embedding_modules = _fetch_all_embeddings(model)
    else:
        embedding_modules = []
        assert isinstance(select_embeddings, List), 'the embedding_modules must be a list of embedding modules'
        for emb in select_embeddings:
            assert type(emb) in SUPPORTED_MODULES, 'the embedding_modules list must be an embedding or embedding bags'
            fqn_name = module_to_fqn(model, emb)
            assert fqn_name is not None, 'the embedding modules must be part of input model'
            embedding_modules.append((fqn_name, emb))
    if sparsify_first:
        for (name, emb_module) in embedding_modules:
            valid_name = name.replace('.', '_')
            data_sparsifier.add_data(name=valid_name, data=emb_module)
        data_sparsifier.step()
        data_sparsifier.squash_mask()
        for (_, emb_module) in embedding_modules:
            emb_module.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
        torch.ao.quantization.prepare(model, inplace=True)
        torch.ao.quantization.convert(model, inplace=True)
    else:
        for (_, emb_module) in embedding_modules:
            emb_module.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
        torch.ao.quantization.prepare(model, inplace=True)
        torch.ao.quantization.convert(model, inplace=True)
        quantize_params: Dict[str, Dict] = {'scales': {}, 'zero_points': {}, 'dequant_weights': {}, 'axis': {}, 'dtype': {}}
        for (name, _) in embedding_modules:
            quantized_emb = fqn_to_module(model, name)
            assert quantized_emb is not None
            quantized_weight = quantized_emb.weight()
            quantize_params['scales'][name] = quantized_weight.q_per_channel_scales()
            quantize_params['zero_points'][name] = quantized_weight.q_per_channel_zero_points()
            quantize_params['dequant_weights'][name] = torch.dequantize(quantized_weight)
            quantize_params['axis'][name] = quantized_weight.q_per_channel_axis()
            quantize_params['dtype'][name] = quantized_weight.dtype
            data_sparsifier.add_data(name=name.replace('.', '_'), data=quantize_params['dequant_weights'][name])
        data_sparsifier.step()
        data_sparsifier.squash_mask()
        for (name, _) in embedding_modules:
            quantized_emb = fqn_to_module(model, name)
            assert quantized_emb is not None
            requantized_vector = torch.quantize_per_channel(quantize_params['dequant_weights'][name], scales=quantize_params['scales'][name], zero_points=quantize_params['zero_points'][name], dtype=quantize_params['dtype'][name], axis=quantize_params['axis'][name])
            quantized_emb.set_weight(requantized_vector)