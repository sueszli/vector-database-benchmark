import sys
import os
import ctypes
from typing import List
from ctypes import c_int, c_long, c_float, c_char_p, c_void_p, c_bool, POINTER, pointer, Structure, Array, c_uint8, c_size_t
import pathlib
from bigdl.llm.utils.utils import get_shared_lib_info
from bigdl.llm.utils.common import invalidInputError

def _load_shared_library(lib_base_name: str):
    if False:
        i = 10
        return i + 15
    (_base_path, _lib_paths) = get_shared_lib_info(lib_base_name=lib_base_name)
    if 'BLOOM_CPP_LIB' in os.environ:
        lib_base_name = os.environ['BLOOM_CPP_LIB']
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]
    if sys.platform == 'win32' and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))
        os.environ['PATH'] = str(_base_path) + ';' + os.environ['PATH']
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path))
            except Exception as e:
                invalidInputError(False, f"Failed to load shared library '{_lib_path}': {e}")
    invalidInputError(False, f"Shared library with base name '{lib_base_name}' not found")
_lib_base_name = 'bloom'
_lib = _load_shared_library(_lib_base_name)

def c_free(p: c_void_p):
    if False:
        print('Hello World!')
    _lib.c_free(p)
_lib.c_free.argtypes = [c_void_p]
_lib.c_free.restype = None

def bloom_load(fname: bytes, n_ctx: c_int, n_threads: c_int) -> c_void_p:
    if False:
        print('Hello World!')
    return _lib.bloom_load(fname, n_ctx, n_threads)
_lib.bloom_load.argtypes = [c_char_p, c_int, c_int]
_lib.bloom_load.restype = c_void_p

def bloom_free(ctx: c_void_p):
    if False:
        i = 10
        return i + 15
    return _lib.bloom_free(ctx)
_lib.bloom_free.argtypes = [c_void_p]
_lib.bloom_free.restype = None

def bloom_run(ctx: c_void_p, seed: c_int, n_threads: c_int, n_batch: c_int, n_predict: c_int, match_str: c_bool, prompt: bytes, buf: bytes) -> c_int:
    if False:
        while True:
            i = 10
    return _lib.bloom_run(ctx, seed, n_threads, n_batch, n_predict, match_str, prompt, buf)
_lib.bloom_run.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_bool, c_char_p, c_char_p]
_lib.bloom_run.restype = c_int

def bloom_tokenize(ctx: c_void_p, prompt: bytes, bos: bool=False) -> List[int]:
    if False:
        for i in range(10):
            print('nop')
    n_tokens = c_int(0)
    c_tokens = _lib.tokenize_api(ctx, prompt, bos, pointer(n_tokens))
    tokens = [c_tokens[i] for i in range(0, n_tokens.value)]
    c_free(c_tokens)
    return tokens
_lib.tokenize_api.argtypes = [c_void_p, c_char_p, c_bool, c_void_p]
_lib.tokenize_api.restype = POINTER(c_int)

def bloom_detokenize(ctx: c_void_p, token_id: c_int) -> str:
    if False:
        print('Hello World!')
    c_chars = _lib.detokenize_api(ctx, token_id)
    s = c_chars.decode('utf-8')
    return s
_lib.detokenize_api.argtypes = [c_void_p, c_int]
_lib.detokenize_api.restype = c_char_p

def bloom_eval(ctx: c_void_p, input_ids: List[int], seed: c_int, n_threads: c_int, n_batch: c_int) -> List[List[float]]:
    if False:
        print('Hello World!')
    length = len(input_ids)
    c_input_ids = (c_int * length)(*input_ids)
    n_logits = c_long(0)
    c_logits = _lib.eval_api(ctx, c_input_ids, length, seed, n_threads, n_batch, pointer(n_logits))
    n_vocab = n_logits.value // length
    assert n_vocab * length == n_logits.value
    logits = [[c_logits[i * n_vocab + j] for j in range(n_vocab)] for i in range(length)]
    return logits
_lib.eval_api.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_void_p]
_lib.eval_api.restype = POINTER(c_float)

def bloom_embed(ctx: c_void_p, input_ids: List[int], seed: c_int, n_threads: c_int, n_batch: c_int) -> List[float]:
    if False:
        print('Hello World!')
    length = len(input_ids)
    c_input_ids = (c_int * length)(*input_ids)
    n_embd = c_long(0)
    c_embeddings = _lib.embed_api(ctx, c_input_ids, length, seed, n_threads, n_batch, pointer(n_embd))
    embeddings = [c_embeddings[i] for i in range(n_embd.value)]
    return embeddings
_lib.embed_api.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_void_p]
_lib.embed_api.restype = POINTER(c_float)

def bloom_forward(ctx: c_void_p, input_ids: List[int], seed: c_int, n_threads: c_int, n_batch: c_int) -> int:
    if False:
        print('Hello World!')
    length = len(input_ids)
    c_input_ids = (c_int * length)(*input_ids)
    token_id = _lib.forward_api(ctx, c_input_ids, length, seed, n_threads, n_batch)
    return token_id
_lib.forward_api.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
_lib.forward_api.restype = c_int