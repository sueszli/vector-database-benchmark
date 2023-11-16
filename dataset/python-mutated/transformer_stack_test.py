import copy
import torch
import pytest
from allennlp.common import Params
from allennlp.common import cached_transformers
from allennlp.modules.transformer import TransformerStack, TransformerLayer
PARAMS_DICT = {'num_hidden_layers': 3, 'hidden_size': 6, 'intermediate_size': 3, 'num_attention_heads': 2, 'attention_dropout': 0.1, 'hidden_dropout': 0.2, 'activation': 'relu'}
SEED = 1234

@pytest.fixture
def params():
    if False:
        while True:
            i = 10
    return Params(copy.deepcopy(PARAMS_DICT))

def test_transformer_stack_from_params(params):
    if False:
        print('Hello World!')
    torch.manual_seed(SEED)
    transformer_stack = TransformerStack.from_params(params)
    modules = dict(transformer_stack.named_modules())
    assert len(modules['layers']) == PARAMS_DICT['num_hidden_layers']
    hidden_states = torch.randn(2, 3, 6)
    attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])
    torch.manual_seed(SEED)
    output = transformer_stack.forward(hidden_states, attention_mask=attention_mask)
    torch.manual_seed(SEED)
    layer_params = copy.deepcopy(PARAMS_DICT)
    num_hidden_layers = layer_params.pop('num_hidden_layers')
    transformer_layer = TransformerLayer(**layer_params)
    transformer_stack_from_layer = TransformerStack(num_hidden_layers, transformer_layer)
    torch.manual_seed(SEED)
    from_layer_output = transformer_stack_from_layer.forward(hidden_states, attention_mask=attention_mask)
    assert torch.allclose(from_layer_output.final_hidden_states, output.final_hidden_states)
    with pytest.raises(AssertionError):
        transformer_stack.forward(torch.randn(2, 3, 6), attention_mask=torch.randn(2, 3), encoder_hidden_states=torch.randn(2, 3, 6))

def test_transformer_stack_with_cross_attention(params):
    if False:
        for i in range(10):
            print('nop')
    params['add_cross_attention'] = True
    transformer_stack = TransformerStack.from_params(params).eval()
    modules = dict(transformer_stack.named_modules())
    assert hasattr(modules['layers.0'], 'cross_attention')
    attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])
    transformer_stack.forward(torch.randn(2, 3, 6), attention_mask=attention_mask, encoder_hidden_states=torch.randn(2, 3, 6))

@pytest.mark.parametrize('pretrained_model_name', ['epwalsh/bert-xsmall-dummy', 'bert-base-cased'])
def test_loading_from_pretrained(pretrained_model_name):
    if False:
        return 10
    transformer_stack = TransformerStack.from_pretrained_module(pretrained_model_name).eval()
    pretrained_module = cached_transformers.get(pretrained_model_name, True).encoder.eval()
    batch_size = 2
    seq_length = 15
    hidden_size = transformer_stack.layers[0].get_output_dim()
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    attention_mask = torch.randint(0, 2, (batch_size, seq_length))
    attention_mask_hf = attention_mask[:, None, None, :]
    attention_mask_hf = (1.0 - attention_mask_hf) * -1000000.0
    torch.manual_seed(SEED)
    output = transformer_stack(hidden_states, attention_mask=attention_mask)
    torch.manual_seed(SEED)
    hf_output = pretrained_module(hidden_states, attention_mask=attention_mask_hf)
    assert torch.allclose(output.final_hidden_states, hf_output[0])

def test_loading_partial_pretrained_weights():
    if False:
        return 10
    TransformerStack.from_pretrained_module('bert-base-cased', num_hidden_layers=3, strict=False)