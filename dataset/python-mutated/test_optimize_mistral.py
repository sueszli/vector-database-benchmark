import os
import pytest
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
mistral_model_path = os.environ.get('MISTRAL_ORIGIN_PATH')
prompt = 'Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun'

@pytest.mark.parametrize('Model, Tokenizer, model_path, prompt', [(AutoModelForCausalLM, AutoTokenizer, mistral_model_path, prompt)])
def test_optimize_model(Model, Tokenizer, model_path, prompt):
    if False:
        print('Hello World!')
    tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    model = Model.from_pretrained(model_path, load_in_4bit=True, optimize_model=False, trust_remote_code=True)
    logits_base_model = model(input_ids).logits
    model = Model.from_pretrained(model_path, load_in_4bit=True, optimize_model=True, trust_remote_code=True)
    logits_optimized_model = model(input_ids).logits
    diff = abs(logits_base_model - logits_optimized_model).flatten()
    assert any(diff) is False
if __name__ == '__main__':
    pytest.main([__file__])