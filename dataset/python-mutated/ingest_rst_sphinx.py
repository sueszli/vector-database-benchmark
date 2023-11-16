import os
import pickle
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
import dotenv
import faiss
import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from sphinx.cmd.build import main as sphinx_main

def convert_rst_to_txt(src_dir, dst_dir):
    if False:
        return 10
    if not os.path.exists(src_dir):
        raise Exception('Source directory does not exist')
    for (root, dirs, files) in os.walk(src_dir):
        for file in files:
            if file.endswith('.rst'):
                src_file = os.path.join(root, file.replace('.rst', ''))
                args = f'. -b text -D extensions=sphinx.ext.autodoc -D master_doc={src_file} -D source_suffix=.rst -C {dst_dir} '
                sphinx_main(args.split())
            elif file.endswith('.md'):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(root, file.replace('.md', '.rst'))
                os.rename(src_file, dst_file)
                args = f'. -b text -D extensions=sphinx.ext.autodoc -D master_doc={dst_file} -D source_suffix=.rst -C {dst_dir} '
                sphinx_main(args.split())

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    if False:
        while True:
            i = 10
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    total_price = num_tokens / 1000 * 0.0004
    return (num_tokens, total_price)

def call_openai_api():
    if False:
        i = 10
        return i + 15
    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    faiss.write_index(store.index, 'docs.index')
    store.index = None
    with open('faiss_store.pkl', 'wb') as f:
        pickle.dump(store, f)

def get_user_permission():
    if False:
        while True:
            i = 10
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
ap = ArgumentParser('Script for training DocsGPT on Sphinx documentation')
ap.add_argument('-i', '--inputs', type=str, default='inputs', help='Directory containing documentation files')
args = ap.parse_args()
dotenv.load_dotenv()
src_dir = args.inputs
dst_dir = 'tmp'
convert_rst_to_txt(src_dir, dst_dir)
ps = list(Path('tmp/' + src_dir).glob('**/*.txt'))
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
shutil.rmtree(dst_dir)