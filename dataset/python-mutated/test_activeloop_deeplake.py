from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAIChat
import random
import deeplake
import os
from dotenv import load_dotenv
load_dotenv()

def test():
    if False:
        print('Hello World!')
    loader = TextLoader('docs/extras/modules/state_of_the_union.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = DeepLake(dataset_path='./my_deeplake/', embedding=embeddings, overwrite=True)
    db.add_documents(docs)
    query = 'What did the president say about Ketanji Brown Jackson'
    docs = db.similarity_search(query)
    db = DeepLake(dataset_path='./my_deeplake/', embedding=embeddings, read_only=True)
    docs = db.similarity_search(query)
    qa = RetrievalQA.from_chain_type(llm=OpenAIChat(model='gpt-3.5-turbo'), chain_type='stuff', retriever=db.as_retriever())
    query = 'What did the president say about Ketanji Brown Jackson'
    qa.run(query)
    for d in docs:
        d.metadata['year'] = random.randint(2012, 2014)
    db = DeepLake.from_documents(docs, embeddings, dataset_path='./my_deeplake/', overwrite=True)
    db.similarity_search('What did the president say about Ketanji Brown Jackson', filter={'metadata': {'year': 2013}})
    db.similarity_search('What did the president say about Ketanji Brown Jackson?', distance_metric='cos')
    db.max_marginal_relevance_search('What did the president say about Ketanji Brown Jackson?')
    db.delete_dataset()
    DeepLake.force_delete_by_path('./my_deeplake')
    username = 'testingacc2'
    dataset_path = f'hub://{username}/langchain_testing_python'
    docs = text_splitter.split_documents(documents)
    token = os.environ['ACTIVELOOP_TOKEN']
    embedding = OpenAIEmbeddings()
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings, overwrite=True)
    db.add_documents(docs)
    query = 'What did the president say about Ketanji Brown Jackson'
    docs = db.similarity_search(query)
    print(docs[0].page_content)
    dataset_path = f'hub://{username}/langchain_testing'
    docs = text_splitter.split_documents(documents)
    embedding = OpenAIEmbeddings()
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings, overwrite=True, runtime={'tensor_db': True})
    db.add_documents(docs)
    search_id = db.vectorstore.dataset.id[0].numpy()
    docs = db.similarity_search(query=None, tql=f"SELECT * WHERE id == '{search_id[0]}'")
    bucket = os.environ['BUCKET']
    dataset_path = f's3://{bucket}/langchain_test'
    embedding = OpenAIEmbeddings()
    db = DeepLake.from_documents(docs, dataset_path=dataset_path, embedding=embeddings, overwrite=True)
    db.vectorstore.summary()
    embeds = db.vectorstore.dataset.embedding.numpy()
    source = f'hub://{username}/langchain_testing'
    destination = f'hub://{username}/langchain_test_copy'
    deeplake.deepcopy(src=source, dest=destination, overwrite=True)
    db = DeepLake(dataset_path=destination, embedding=embeddings)
    db.add_documents(docs)