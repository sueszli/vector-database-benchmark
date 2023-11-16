"""Utils module for calculating embeddings for text."""
import sys
import warnings
from itertools import islice
from typing import Optional
import numpy as np
from tqdm import tqdm
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_DIM = 1536
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

def batched(iterable, n):
    if False:
        i = 10
        return i + 15
    'Batch data into tuples of length n. The last batch may be shorter.'
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            return
        yield batch

def encode_text(text, encoding_name):
    if False:
        while True:
            i = 10
    'Encode tokens with the given encoding.'
    if sys.version_info >= (3, 8):
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return encoding.encode(text)
    else:
        return text

def iterate_batched(tokenized_text, chunk_length):
    if False:
        i = 10
        return i + 15
    'Chunk text into tokens of length chunk_length.'
    chunks_iterator = batched(tokenized_text, chunk_length)
    yield from chunks_iterator

def calculate_builtin_embeddings(text: np.array, model: str='miniLM', file_path: Optional[str]='embeddings.npy', device: Optional[str]=None, long_sample_behaviour: str='average+warn', open_ai_batch_size: int=500) -> np.array:
    if False:
        while True:
            i = 10
    "\n    Get the built-in embeddings for the dataset.\n\n    Parameters\n    ----------\n    text : np.array\n        The text to get embeddings for.\n    model : str, default 'miniLM'\n        The type of embeddings to return. Can be either 'miniLM' or 'open_ai'.\n        For 'open_ai' option, the model used is 'text-embedding-ada-002' and requires to first set an open ai api key\n        by using the command openai.api_key = YOUR_API_KEY\n    file_path : Optional[str], default 'embeddings.csv'\n        If given, the embeddings will be saved to the given file path.\n    device : str, default None\n        The device to use for the embeddings. If None, the default device will be used.\n    long_sample_behaviour : str, default 'average+warn'\n        How to handle long samples. Averaging is done as described in\n        https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb\n        Currently, applies only to the 'open_ai' model, as the 'miniLM' model can handle long samples.\n\n        Options are:\n            - 'average+warn' (default): average the embeddings of the chunks and warn if the sample is too long.\n            - 'average': average the embeddings of the chunks.\n            - 'truncate': truncate the sample to the maximum length.\n            - 'raise': raise an error if the sample is too long.\n            - 'nan': return an embedding vector of nans for each sample that is too long.\n    open_ai_batch_size : int, default 500\n        The amount of samples to send to open ai in each batch. Reduce if getting errors from open ai.\n\n    Returns\n    -------\n        np.array\n            The embeddings for the dataset.\n    "
    if model == 'miniLM':
        try:
            import sentence_transformers
        except ImportError as e:
            raise ImportError('calculate_builtin_embeddings with model="miniLM" requires the sentence_transformers python package. To get it, run "pip install sentence_transformers".') from e
        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2', device=device)
        embeddings = model.encode(text)
    elif model == 'open_ai':
        try:
            import openai
        except ImportError as e:
            raise ImportError('calculate_builtin_embeddings with model="open_ai" requires the openai python package. To get it, run "pip install openai".') from e
        from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

        @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), retry=retry_if_not_exception_type(openai.InvalidRequestError))
        def _get_embedding_with_backoff(text_or_tokens, model=EMBEDDING_MODEL):
            if False:
                print('Hello World!')
            return openai.Embedding.create(input=text_or_tokens, model=model)['data']

        def len_safe_get_embedding(list_of_texts, model_name=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING):
            if False:
                i = 10
                return i + 15
            'Get embeddings for a list of texts, chunking them if necessary.'
            chunked_texts = []
            chunk_lens = []
            encoded_texts = []
            max_sample_length = 0
            skip_sample_indices = set()
            for (i, text_sample) in enumerate(list_of_texts):
                tokens_in_sample = encode_text(text_sample, encoding_name=encoding_name)
                tokens_per_sample = []
                num_chunks = 0
                for chunk in iterate_batched(tokens_in_sample, chunk_length=max_tokens):
                    if long_sample_behaviour == 'nan' and num_chunks > 0:
                        skip_sample_indices.add(i)
                        break
                    chunked_texts.append((i, chunk))
                    chunk_lens.append(len(chunk))
                    tokens_per_sample += chunk
                    max_sample_length = max(max_sample_length, len(tokens_per_sample))
                    num_chunks += 1
                    if long_sample_behaviour == 'truncate':
                        break
                encoded_texts.append(tokens_per_sample)
            if max_sample_length > max_tokens:
                if long_sample_behaviour == 'average+warn':
                    warnings.warn(f'At least one sample is longer than {max_tokens} tokens, which is the maximum context window handled by {model}. Maximal sample length found is {max_sample_length} tokens. The sample will be split into chunks and the embeddings will be averaged. To avoid this warning, set long_sample_behaviour="average" or long_sample_behaviour="truncate".')
                elif long_sample_behaviour == 'raise':
                    raise ValueError(f'At least one sample is longer than {max_tokens} tokens, which is the maximum context window handled by {model}. Maximal sample length found is {max_sample_length} tokens. To avoid this error, set long_sample_behaviour="average" or long_sample_behaviour="truncate".')
            filtered_chunked_texts = [chunk for (i, chunk) in chunked_texts if i not in skip_sample_indices]
            chunk_embeddings_output = []
            for sub_list in tqdm([filtered_chunked_texts[x:x + open_ai_batch_size] for x in range(0, len(filtered_chunked_texts), open_ai_batch_size)], desc='Calculating Embeddings '):
                chunk_embeddings_output.extend(_get_embedding_with_backoff(sub_list, model=model_name))
            chunk_embeddings = [embedding['embedding'] for embedding in chunk_embeddings_output]
            result_embeddings = []
            idx = 0
            for (i, tokens_in_sample) in enumerate(encoded_texts):
                if i in skip_sample_indices:
                    text_embedding = np.ones((EMBEDDING_DIM,)) * np.nan
                else:
                    text_embeddings = []
                    text_lens = []
                    while idx < len(chunk_lens) and sum(text_lens) < len(tokens_in_sample):
                        text_embeddings.append(chunk_embeddings[idx])
                        text_lens.append(chunk_lens[idx])
                        idx += 1
                    if sum(text_lens) == 0:
                        text_embedding = np.ones((EMBEDDING_DIM,)) * np.nan
                    else:
                        text_embedding = np.average(text_embeddings, axis=0, weights=text_lens)
                        text_embedding = text_embedding / np.linalg.norm(text_embedding)
                result_embeddings.append(text_embedding.tolist())
            return result_embeddings
        clean_text = [_clean_special_chars(x) for x in text]
        embeddings = len_safe_get_embedding(clean_text)
    else:
        raise ValueError(f'Unknown model type: {model}')
    embeddings = np.array(embeddings).astype(np.float16)
    if file_path is not None:
        np.save(file_path, embeddings)
    return embeddings

def _clean_special_chars(text):
    if False:
        return 10
    special_chars = '!@#$%^&*()_+{}|:"<>?~`-=[]\\;\\\',./'
    for char in special_chars:
        text = text.replace(char, '')
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.replace('<br />', ' ')
    return text