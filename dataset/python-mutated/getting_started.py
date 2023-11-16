from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import build_pipeline, add_example_data, print_answers

def getting_started(provider, API_KEY):
    if False:
        print('Hello World!')
    '\n    This getting_started example shows you how to use LLMs with your data with a technique called Retrieval Augmented Generation - RAG.\n\n    :param provider: We are model agnostic :) Here, you can choose from: "anthropic", "cohere", "huggingface", and "openai".\n    :param API_KEY: The API key matching the provider.\n\n    '
    document_store = InMemoryDocumentStore(use_bm25=True)
    pipeline = build_pipeline(provider, API_KEY, document_store)
    add_example_data(document_store, 'data/GoT_getting_started')
    result = pipeline.run(query='Who is the father of Arya Stark?')
    print_answers(result, details='medium')
    return result
if __name__ == '__main__':
    getting_started(provider='openai', API_KEY='ADD KEY HERE')