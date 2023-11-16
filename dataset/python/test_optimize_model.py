#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import pytest

import torch
from transformers import LlamaTokenizer, AutoTokenizer
from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel


device = os.environ['DEVICE']
print(f'Running on {device}')
if device == 'xpu':
    import intel_extension_for_pytorch as ipex

prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

@pytest.mark.parametrize('Model, Tokenizer, model_path',[
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('MPT_7B_ORIGIN_PATH')),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('FALCON_7B_ORIGIN_PATH'))
    ])
def test_optimize_model(Model, Tokenizer, model_path):
    tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    model = Model.from_pretrained(model_path,
                                load_in_4bit=True,
                                optimize_model=False,
                                trust_remote_code=True)
    model = model.to(device)
    logits_base_model = (model(input_ids)).logits
    model.to('cpu')  # deallocate gpu memory

    model = Model.from_pretrained(model_path,
                                load_in_4bit=True,
                                optimize_model=True,
                                trust_remote_code=True)
    model = model.to(device)
    logits_optimized_model = (model(input_ids)).logits
    model.to('cpu')
    
    diff = abs(logits_base_model - logits_optimized_model).flatten()

    assert any(diff) is False

class Test_Optimize_Gpu_Model:
    def setup(self):

        self.layer_outputs = []
        self.pre_layer_outputs = []

    def run_optimize_gpu_model(self, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound):
        def forward_hook(module, input, output, layer_name):
            self.layer_outputs.append(output)

        def pre_forward_hook(module, input, output, layer_name):
            self.pre_layer_outputs.append(output)

        tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        model = Model.from_pretrained(model_path,
                                      load_in_4bit=True,
                                      optimize_model=False,
                                      trust_remote_code=True)
        model = model.to(device)

        for layer_name, layer_module in model.named_modules():
            if layer_name == layer_norm:
                layer_module.register_forward_hook(
                    lambda module, input, output, layer_name=layer_name: pre_forward_hook(module, input,
                                                                                          output, layer_name))
            if layer_name == self_attn:
                layer_module.register_forward_hook(
                    lambda module, input, output, layer_name=layer_name: forward_hook(module, input,
                                                                                      output, layer_name))
        logits_base_model = (model(input_ids)).logits
        # the list `layer_output` has only one element.
        layer_tensor = self.layer_outputs.pop()
        model.to('cpu')

        opt_model = Model.from_pretrained(model_path,
                                          load_in_4bit=True,
                                          optimize_model=True,
                                          trust_remote_code=True)
        opt_model = opt_model.to(device)


        def replace_forward_hook(module, input, output, layer_name):
            output = self.pre_layer_outputs[0]
            return output

        for layer_name, layer_module in opt_model.named_modules():
            if layer_name == layer_norm:
                layer_module.register_forward_hook(
                    lambda module, input, output, layer_name=layer_name: replace_forward_hook(module, input,
                                                                                              output, layer_name))
            if layer_name == self_attn:
                layer_module.register_forward_hook(
                    lambda module, input, output, layer_name=layer_name: forward_hook(module, input,
                                                                                      output, layer_name))
        logits_optimized_model = (opt_model(input_ids)).logits
        # the list `layer_output` has only one element.
        opt_layer_tensor = self.layer_outputs[0]
        opt_model.to('cpu')

        attn_output_diff = []
        for i, (t1, t2) in enumerate(zip(layer_tensor, opt_layer_tensor)):
            if t1 is not None and t2 is not None:
                if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                    # 'attn_output' is of type torch.Tensor.
                    attn_output_diff.append(t1 - t2)
                else:
                    # 'past_key_value'is of type tuple as default.
                    for i, (t3, t4) in enumerate(zip(t1, t2)):
                        if model.config.architectures[0] == "ChatGLMModel" and \
                                hasattr(model.config, 'padded_vocab_size') and \
                                model.config.padded_vocab_size == 65024:
                            # chatglm2's past_key_value is expanded 16x for some speedup.
                            # We need to narrow it here.
                            t4 = t4[:, :, 15:17, :]
                        attn_output_diff.append(t3 - t4)

        max_diff_tensor = [torch.max(item).item() for item in attn_output_diff]
        print(max_diff_tensor)
        assert all(max_diff <= lower_bound for max_diff in max_diff_tensor)


    def test_falcon_gpu_model(self):

        Model = AutoModelForCausalLM
        Tokenizer = AutoTokenizer
        model_path = os.environ.get('FALCON_7B_ORIGIN_PATH')
        # currently only compare the output of the last self-attention layer.
        layer_norm = "transformer.h.31.input_layernorm"
        self_attn = "transformer.h.31.self_attention"
        lower_bound = 0

        self.run_optimize_gpu_model(Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)


    def test_llama_gpu_model(self):

        Model = AutoModelForCausalLM
        Tokenizer = AutoTokenizer
        model_path = os.environ.get('LLAMA2_7B_ORIGIN_PATH')
        # currently only compare the output of the last self-attention layer.
        layer_norm = "model.layers.31.input_layernorm"
        self_attn = "model.layers.31.self_attn"
        lower_bound = 5e-2

        self.run_optimize_gpu_model(Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)

    def test_chatglm2_gpu_model(self):

        Model = AutoModel
        Tokenizer = AutoTokenizer
        model_path = os.environ.get('CHATGLM2_6B_ORIGIN_PATH')
        # currently only need to compare the output of one self-attention layer.
        layer_norm = "transformer.encoder.layers.27.input_layernorm"
        self_attn = "transformer.encoder.layers.27.self_attention"
        lower_bound = 5e-5

        self.run_optimize_gpu_model(Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)


if __name__ == '__main__':
    pytest.main([__file__])
