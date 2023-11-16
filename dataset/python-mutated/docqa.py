import argparse
from langchain.vectorstores import Chroma
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from bigdl.llm.langchain.llms import *
from bigdl.llm.langchain.embeddings import *

def main(args):
    if False:
        while True:
            i = 10
    input_path = args.input_path
    model_path = args.model_path
    model_family = args.model_family
    query = args.question
    n_ctx = args.n_ctx
    n_threads = args.thread_num
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    with open(input_path) as f:
        input_doc = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(input_doc)
    model_family_to_embeddings = {'llama': LlamaEmbeddings, 'gptneox': GptneoxEmbeddings, 'bloom': BloomEmbeddings, 'starcoder': StarcoderEmbeddings}
    model_family_to_llm = {'llama': LlamaLLM, 'gptneox': GptneoxLLM, 'bloom': BloomLLM, 'starcoder': StarcoderLLM}
    if model_family in model_family_to_embeddings and model_family in model_family_to_llm:
        llm_embeddings = model_family_to_embeddings[model_family]
        langchain_llm = model_family_to_llm[model_family]
    else:
        raise ValueError(f'Unknown model family: {model_family}')
    embeddings = llm_embeddings(model_path=model_path, n_threads=n_threads, n_ctx=n_ctx)
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{'source': str(i)} for i in range(len(texts))]).as_retriever()
    docs = docsearch.get_relevant_documents(query)
    bigdl_llm = langchain_llm(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, callback_manager=callback_manager)
    doc_chain = load_qa_chain(bigdl_llm, chain_type='stuff', prompt=QA_PROMPT, callback_manager=callback_manager)
    doc_chain.run(input_documents=docs, question=query)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BigDLCausalLM Langchain QA over Docs Example')
    parser.add_argument('-x', '--model-family', type=str, required=True, choices=['llama', 'bloom', 'gptneox', 'chatglm', 'starcoder'], help='the model family')
    parser.add_argument('-m', '--model-path', type=str, required=True, help='the path to the converted llm model')
    parser.add_argument('-i', '--input-path', type=str, required=True, help='the path to the input doc.')
    parser.add_argument('-q', '--question', type=str, default='What is AI?', help='qustion you want to ask.')
    parser.add_argument('-c', '--n-ctx', type=int, default=2048, help='the maximum context size')
    parser.add_argument('-t', '--thread-num', type=int, default=2, help='number of threads to use for inference')
    args = parser.parse_args()
    main(args)