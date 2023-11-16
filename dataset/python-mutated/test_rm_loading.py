from argparse import Namespace
import model_training.models.reward_model
from model_training.models.reward_model import GPTNeoXRewardModel
from model_training.utils.utils import get_tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def test_convert_model(model_name: str='EleutherAI/pythia-70m-deduped', cache_dir: str='.cache', output_dir: str='.saved_models_rm/debug'):
    if False:
        return 10
    training_conf = Namespace(cache_dir=cache_dir, model_name=model_name)
    tokenizer = get_tokenizer(training_conf)
    model = GPTNeoXRewardModel.from_pretrained(model_name, cache_dir=cache_dir)
    print('model', type(model))
    print('tokenizer', type(tokenizer))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def test_load_reward_model(model_name: str='andreaskoepf/oasst-rm-1-pythia-1b', cache_dir: str='.cache'):
    if False:
        i = 10
        return i + 15
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    rm = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    print('auto', type(rm))
    print('auto.config', type(rm.config))
    question = '<|prompter|>Hi how are you?<|endoftext|><|assistant|>Hi, I am Open-Assistant a large open-source language model trained by LAION AI. How can I help you today?<|endoftext|>'
    inputs = tokenizer(question, return_tensors='pt')
    print(inputs)
    score = rm(**inputs).logits[0].cpu().detach()
    print(score)
if __name__ == '__main__':
    test_load_reward_model('andreaskoepf/oasst-rm-1-pythia-1b')