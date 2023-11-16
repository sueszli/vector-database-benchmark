import time
import argparse
from bigdl.llm.transformers import *

def convert(repo_id_or_model_path, model_family, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    from bigdl.llm import llm_convert
    original_llm_path = repo_id_or_model_path
    bigdl_llm_path = llm_convert(model=original_llm_path, outfile='./', outtype='int4', tmp_path=tmp_path, model_family=model_family)
    return bigdl_llm_path

def load(model_path, model_family, n_threads):
    if False:
        return 10
    model_family_to_class = {'llama': LlamaForCausalLM, 'gptneox': GptneoxForCausalLM, 'bloom': BloomForCausalLM, 'starcoder': StarcoderForCausalLM, 'chatglm': ChatGLMForCausalLM}
    if model_family in model_family_to_class:
        llm_causal = model_family_to_class[model_family]
    else:
        raise ValueError(f'Unknown model family: {model_family}')
    llm = llm_causal.from_pretrained(pretrained_model_name_or_path=model_path, native=True, dtype='int4', n_threads=n_threads)
    return llm

def inference(llm, repo_id_or_model_path, model_family, prompt):
    if False:
        print('Hello World!')
    if model_family in ['llama', 'gptneox', 'bloom', 'starcoder', 'chatglm']:
        print('-' * 20, ' bigdl-llm based tokenizer ', '-' * 20)
        st = time.time()
        tokens_id = llm.tokenize(prompt)
        output_tokens_id = llm.generate(tokens_id, max_new_tokens=32)
        output = llm.batch_decode(output_tokens_id)
        print(f'Inference time: {time.time() - st} s')
        print(f'Output:\n{output}')
        print('-' * 20, ' HuggingFace transformers tokenizer ', '-' * 20)
        print('Please note that the loading of HuggingFace transformers tokenizer may take some time.\n')
        if model_family == 'llama':
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(repo_id_or_model_path)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(repo_id_or_model_path)
        st = time.time()
        tokens_id = tokenizer(prompt).input_ids
        output_tokens_id = llm.generate(tokens_id, max_new_tokens=32)
        output = tokenizer.batch_decode(output_tokens_id)
        print(f'Inference time: {time.time() - st} s')
        print(f'Output:\n{output}')
        print('-' * 20, ' fast forward ', '-' * 20)
        st = time.time()
        output = llm(prompt, max_tokens=32)
        print(f'Inference time (fast forward): {time.time() - st} s')
        print(f'Output:\n{output}')

def main():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='INT4 pipeline example')
    parser.add_argument('--thread-num', type=int, default=2, required=True, help='Number of threads to use for inference')
    parser.add_argument('--model-family', type=str, default='llama', required=True, choices=['llama', 'llama2', 'bloom', 'gptneox', 'starcoder', 'chatglm'], help="The model family of the large language model (supported option: 'llama', 'llama2', 'gptneox', 'bloom', 'starcoder', 'chatglm')")
    parser.add_argument('--repo-id-or-model-path', type=str, required=True, help='The path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default='Once upon a time, there existed a little girl who liked to have adventures. ', help='Prompt to infer')
    parser.add_argument('--tmp-path', type=str, default='/tmp', help='path to store intermediate model during the conversion process')
    args = parser.parse_args()
    repo_id_or_model_path = args.repo_id_or_model_path
    if args.model_family == 'llama2':
        args.model_family = 'llama'
    bigdl_llm_path = convert(repo_id_or_model_path=repo_id_or_model_path, model_family=args.model_family, tmp_path=args.tmp_path)
    llm = load(model_path=bigdl_llm_path, model_family=args.model_family, n_threads=args.thread_num)
    inference(llm=llm, repo_id_or_model_path=repo_id_or_model_path, model_family=args.model_family, prompt=args.prompt)
if __name__ == '__main__':
    main()