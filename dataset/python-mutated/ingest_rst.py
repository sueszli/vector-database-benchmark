import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
import dotenv
import faiss
import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    if False:
        print('Hello World!')
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    total_price = num_tokens / 1000 * 0.0004
    return (num_tokens, total_price)

def call_openai_api():
    if False:
        while True:
            i = 10
    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    faiss.write_index(store.index, 'docs.index')
    store.index = None
    with open('faiss_store.pkl', 'wb') as f:
        pickle.dump(store, f)

def get_user_permission():
    if False:
        i = 10
        return i + 15
    docs_content = ' '.join(docs)
    (tokens, total_price) = num_tokens_from_string(string=docs_content, encoding_name='cl100k_base')
    print(f"Number of Tokens = {format(tokens, ',d')}")
    print(f"Approx Cost = ${format(total_price, ',.2f')}")
    user_input = input('Price Okay? (Y/N) \n').lower()
    if user_input == 'y':
        call_openai_api()
    elif user_input == '':
        call_openai_api()
    else:
        print('The API was not called. No money was spent.')
dotenv.load_dotenv()
ap = ArgumentParser('Script for training DocsGPT on .rst documentation files.')
ap.add_argument('-i', '--inputs', type=str, default='inputs', help='Directory containing documentation files')
args = ap.parse_args()
ps = list(Path(args.inputs).glob('**/*.rst'))
data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p)
text_splitter = CharacterTextSplitter(chunk_size=1500, separator='\n')
docs = []
metadatas = []
for (i, d) in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{'source': sources[i]}] * len(splits))
if len(sys.argv) > 1:
    permission_bypass_flag = sys.argv[1]
    if permission_bypass_flag == '-y':
        call_openai_api()
    else:
        get_user_permission()
else:
    get_user_permission()