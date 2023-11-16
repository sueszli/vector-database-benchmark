import argparse
from langchain.chains import LLMMathChain
from bigdl.llm.langchain.llms import TransformersLLM, TransformersPipelineLLM

def main(args):
    if False:
        print('Hello World!')
    question = args.question
    model_path = args.model_path
    llm = TransformersLLM.from_model_id(model_id=model_path, model_kwargs={'temperature': 0, 'max_length': 1024, 'trust_remote_code': True})
    llm_math = LLMMathChain.from_llm(llm, verbose=True)
    output = llm_math.run(question)
    print('====output=====')
    print(output)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransformersLLM Langchain Math Example')
    parser.add_argument('-m', '--model-path', type=str, required=True, help='the path to transformers model')
    parser.add_argument('-q', '--question', type=str, default='What is 13 raised to the .3432 power?', help='qustion you want to ask.')
    args = parser.parse_args()
    main(args)