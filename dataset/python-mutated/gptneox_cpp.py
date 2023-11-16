import sys
import os
import ctypes
from ctypes import c_int, c_float, c_char_p, c_void_p, c_bool, pointer, POINTER, _Pointer, Structure, Array, c_uint8, c_size_t
import pathlib
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.utils.utils import get_shared_lib_info

def _load_shared_library(lib_base_name: str):
    if False:
        for i in range(10):
            print('nop')
    (_base_path, _lib_paths) = get_shared_lib_info(lib_base_name=lib_base_name)
    if 'GPTNEOX_CPP_LIB' in os.environ:
        lib_base_name = os.environ['GPTNEOX_CPP_LIB']
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]
    cdll_args = dict()
    if sys.platform == 'win32' and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))
        os.environ['PATH'] = str(_base_path) + ';' + os.environ['PATH']
        cdll_args['winmode'] = 0
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)
            except Exception as e:
                invalidInputError(False, f"Failed to load shared library '{_lib_path}': {e}.")
    invalidInputError(False, f"Shared library with base name '{lib_base_name}' not found.")
_lib_base_name = 'gptneox'
_lib = _load_shared_library(_lib_base_name)
GPTNEOX_FILE_VERSION = c_int(1)
GPTNEOX_FILE_MAGIC = b'ggjt'
GPTNEOX_FILE_MAGIC_UNVERSIONED = b'ggml'
gptneox_context_p = c_void_p
gptneox_token = c_int
gptneox_token_p = POINTER(gptneox_token)

class gptneox_token_data(Structure):
    _fields_ = [('id', gptneox_token), ('logit', c_float), ('p', c_float)]
gptneox_token_data_p = POINTER(gptneox_token_data)

class gptneox_token_data_array(Structure):
    _fields_ = [('data', gptneox_token_data_p), ('size', c_size_t), ('sorted', c_bool)]
gptneox_token_data_array_p = POINTER(gptneox_token_data_array)
gptneox_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)

class gptneox_context_params(Structure):
    _fields_ = [('n_ctx', c_int), ('n_parts', c_int), ('seed', c_int), ('f16_kv', c_bool), ('logits_all', c_bool), ('vocab_only', c_bool), ('use_mmap', c_bool), ('use_mlock', c_bool), ('embedding', c_bool), ('progress_callback', gptneox_progress_callback), ('progress_callback_user_data', c_void_p)]
gptneox_context_params_p = POINTER(gptneox_context_params)
GPTNEOX_FTYPE_ALL_F32 = c_int(0)
GPTNEOX_FTYPE_MOSTLY_F16 = c_int(1)
GPTNEOX_FTYPE_MOSTLY_Q4_0 = c_int(2)
GPTNEOX_FTYPE_MOSTLY_Q4_1 = c_int(3)
GPTNEOX_FTYPE_MOSTLY_Q4_1_SOME_F16 = c_int(4)
GPTNEOX_FTYPE_MOSTLY_Q4_2 = c_int(5)
GPTNEOX_FTYPE_MOSTLY_Q8_0 = c_int(7)
GPTNEOX_FTYPE_MOSTLY_Q5_0 = c_int(8)
GPTNEOX_FTYPE_MOSTLY_Q5_1 = c_int(9)
c_float_p = POINTER(c_float)
c_uint8_p = POINTER(c_uint8)
c_size_t_p = POINTER(c_size_t)

def gptneox_context_default_params() -> gptneox_context_params:
    if False:
        print('Hello World!')
    return _lib.gptneox_context_default_params()
_lib.gptneox_context_default_params.argtypes = []
_lib.gptneox_context_default_params.restype = gptneox_context_params

def gptneox_mmap_supported() -> bool:
    if False:
        return 10
    return _lib.gptneox_mmap_supported()
_lib.gptneox_mmap_supported.argtypes = []
_lib.gptneox_mmap_supported.restype = c_bool

def gptneox_mlock_supported() -> bool:
    if False:
        for i in range(10):
            print('nop')
    return _lib.gptneox_mlock_supported()
_lib.gptneox_mlock_supported.argtypes = []
_lib.gptneox_mlock_supported.restype = c_bool

def gptneox_init_from_file(path_model: bytes, params: gptneox_context_params) -> gptneox_context_p:
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_init_from_file(path_model, pointer(params))
_lib.gptneox_init_from_file.argtypes = [c_char_p, gptneox_context_params_p]
_lib.gptneox_init_from_file.restype = gptneox_context_p

def gptneox_free(ctx: gptneox_context_p):
    if False:
        i = 10
        return i + 15
    _lib.gptneox_free(ctx)
_lib.gptneox_free.argtypes = [gptneox_context_p]
_lib.gptneox_free.restype = None

def gptneox_model_quantize(fname_inp: bytes, fname_out: bytes, ftype: c_int, nthread: c_int) -> c_int:
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_model_quantize(fname_inp, fname_out, ftype, nthread)
_lib.gptneox_model_quantize.argtypes = [c_char_p, c_char_p, c_int, c_int]
_lib.gptneox_model_quantize.restype = c_int

def gptneox_model_copy(fname_inp: bytes, fname_out: bytes, ftype: c_int) -> c_int:
    if False:
        for i in range(10):
            print('nop')
    return _lib.gptneox_model_copy(fname_inp, fname_out, ftype)
_lib.gptneox_model_copy.argtypes = [c_char_p, c_char_p, c_int]
_lib.gptneox_model_copy.restype = c_int

def gptneox_apply_lora_from_file(ctx: gptneox_context_p, path_lora: c_char_p, path_base_model: c_char_p, n_threads: c_int) -> c_int:
    if False:
        for i in range(10):
            print('nop')
    return _lib.gptneox_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads)
_lib.gptneox_apply_lora_from_file.argtypes = [gptneox_context_p, c_char_p, c_char_p, c_int]
_lib.gptneox_apply_lora_from_file.restype = c_int

def gptneox_get_kv_cache_token_count(ctx: gptneox_context_p) -> c_int:
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_get_kv_cache_token_count(ctx)
_lib.gptneox_get_kv_cache_token_count.argtypes = [gptneox_context_p]
_lib.gptneox_get_kv_cache_token_count.restype = c_int

def gptneox_set_rng_seed(ctx: gptneox_context_p, seed: c_int):
    if False:
        while True:
            i = 10
    return _lib.gptneox_set_rng_seed(ctx, seed)
_lib.gptneox_set_rng_seed.argtypes = [gptneox_context_p, c_int]
_lib.gptneox_set_rng_seed.restype = None

def gptneox_get_state_size(ctx: gptneox_context_p) -> c_size_t:
    if False:
        return 10
    return _lib.gptneox_get_state_size(ctx)
_lib.gptneox_get_state_size.argtypes = [gptneox_context_p]
_lib.gptneox_get_state_size.restype = c_size_t

def gptneox_copy_state_data(ctx: gptneox_context_p, dst) -> int:
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_copy_state_data(ctx, dst)
_lib.gptneox_copy_state_data.argtypes = [gptneox_context_p, c_uint8_p]
_lib.gptneox_copy_state_data.restype = c_size_t

def gptneox_set_state_data(ctx: gptneox_context_p, src) -> int:
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_set_state_data(ctx, src)
_lib.gptneox_set_state_data.argtypes = [gptneox_context_p, c_uint8_p]
_lib.gptneox_set_state_data.restype = c_size_t

def gptneox_load_session_file(ctx: gptneox_context_p, path_session: bytes, tokens_out, n_token_capacity: c_size_t, n_token_count_out) -> c_size_t:
    if False:
        return 10
    return _lib.gptneox_load_session_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)
_lib.gptneox_load_session_file.argtypes = [gptneox_context_p, c_char_p, gptneox_token_p, c_size_t, c_size_t_p]
_lib.gptneox_load_session_file.restype = c_size_t

def gptneox_save_session_file(ctx: gptneox_context_p, path_session: bytes, tokens, n_token_count: c_size_t) -> c_size_t:
    if False:
        while True:
            i = 10
    return _lib.gptneox_save_session_file(ctx, path_session, tokens, n_token_count)
_lib.gptneox_save_session_file.argtypes = [gptneox_context_p, c_char_p, gptneox_token_p, c_size_t]
_lib.gptneox_save_session_file.restype = c_size_t

def gptneox_eval(ctx: gptneox_context_p, tokens, n_tokens: c_int, n_past: c_int, n_threads: c_int) -> c_int:
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_eval(ctx, tokens, n_tokens, n_past, n_threads)
_lib.gptneox_eval.argtypes = [gptneox_context_p, gptneox_token_p, c_int, c_int, c_int]
_lib.gptneox_eval.restype = c_int

def gptneox_tokenize(ctx: gptneox_context_p, text: bytes, tokens, n_max_tokens: c_int, add_bos: c_bool) -> int:
    if False:
        print('Hello World!')
    return _lib.gptneox_tokenize(ctx, text, tokens, n_max_tokens, add_bos)
_lib.gptneox_tokenize.argtypes = [gptneox_context_p, c_char_p, gptneox_token_p, c_int, c_bool]
_lib.gptneox_tokenize.restype = c_int

def gptneox_n_vocab(ctx: gptneox_context_p) -> c_int:
    if False:
        return 10
    return _lib.gptneox_n_vocab(ctx)
_lib.gptneox_n_vocab.argtypes = [gptneox_context_p]
_lib.gptneox_n_vocab.restype = c_int

def gptneox_n_ctx(ctx: gptneox_context_p) -> c_int:
    if False:
        while True:
            i = 10
    return _lib.gptneox_n_ctx(ctx)
_lib.gptneox_n_ctx.argtypes = [gptneox_context_p]
_lib.gptneox_n_ctx.restype = c_int

def gptneox_n_embd(ctx: gptneox_context_p) -> c_int:
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_n_embd(ctx)
_lib.gptneox_n_embd.argtypes = [gptneox_context_p]
_lib.gptneox_n_embd.restype = c_int

def gptneox_get_logits(ctx: gptneox_context_p):
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_get_logits(ctx)
_lib.gptneox_get_logits.argtypes = [gptneox_context_p]
_lib.gptneox_get_logits.restype = c_float_p

def gptneox_get_embeddings(ctx: gptneox_context_p):
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_get_embeddings(ctx)
_lib.gptneox_get_embeddings.argtypes = [gptneox_context_p]
_lib.gptneox_get_embeddings.restype = c_float_p

def gptneox_token_to_str(ctx: gptneox_context_p, token: gptneox_token) -> bytes:
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_token_to_str(ctx, token)
_lib.gptneox_token_to_str.argtypes = [gptneox_context_p, gptneox_token]
_lib.gptneox_token_to_str.restype = c_char_p

def gptneox_str_to_token(ctx: gptneox_context_p, input_str: c_char_p):
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_str_to_token(ctx, input_str)
_lib.gptneox_str_to_token.argtypes = [gptneox_context_p, c_char_p]
_lib.gptneox_str_to_token.restype = gptneox_token

def gptneox_token_bos() -> gptneox_token:
    if False:
        print('Hello World!')
    return _lib.gptneox_token_bos()
_lib.gptneox_token_bos.argtypes = []
_lib.gptneox_token_bos.restype = gptneox_token

def gptneox_token_eos() -> gptneox_token:
    if False:
        while True:
            i = 10
    return _lib.gptneox_token_eos()
_lib.gptneox_token_eos.argtypes = []
_lib.gptneox_token_eos.restype = gptneox_token

def gptneox_get_candidates(ctx: gptneox_context_p, n_vocab: c_int, logits: c_float_p):
    if False:
        for i in range(10):
            print('nop')
    return _lib.gptneox_get_candidates(ctx, n_vocab, logits)
_lib.gptneox_get_candidates.argtypes = [gptneox_context_p, c_int, c_float_p]
_lib.gptneox_get_candidates.restype = gptneox_token_data_array

def gptneox_sample_repetition_penalty(ctx: gptneox_context_p, candidates, last_tokens_data, last_tokens_size: c_int, penalty: c_float):
    if False:
        while True:
            i = 10
    return _lib.gptneox_sample_repetition_penalty(ctx, candidates, last_tokens_data, last_tokens_size, penalty)
_lib.gptneox_sample_repetition_penalty.argtypes = [gptneox_context_p, gptneox_token_data_array_p, gptneox_token_p, c_int, c_float]
_lib.gptneox_sample_repetition_penalty.restype = None

def gptneox_sample_frequency_and_presence_penalties(ctx: gptneox_context_p, candidates, last_tokens_data, last_tokens_size: c_int, alpha_frequency: c_float, alpha_presence: c_float):
    if False:
        return 10
    return _lib.gptneox_sample_frequency_and_presence_penalties(ctx, candidates, last_tokens_data, last_tokens_size, alpha_frequency, alpha_presence)
_lib.gptneox_sample_frequency_and_presence_penalties.argtypes = [gptneox_context_p, gptneox_token_data_array_p, gptneox_token_p, c_int, c_float, c_float]
_lib.gptneox_sample_frequency_and_presence_penalties.restype = None

def gptneox_sample_softmax(ctx: gptneox_context_p, candidates):
    if False:
        for i in range(10):
            print('nop')
    return _lib.gptneox_sample_softmax(ctx, candidates)
_lib.gptneox_sample_softmax.argtypes = [gptneox_context_p, gptneox_token_data_array_p]
_lib.gptneox_sample_softmax.restype = None

def gptneox_sample_top_k(ctx: gptneox_context_p, candidates, k: c_int, min_keep: c_size_t):
    if False:
        print('Hello World!')
    return _lib.gptneox_sample_top_k(ctx, candidates, k, min_keep)
_lib.gptneox_sample_top_k.argtypes = [gptneox_context_p, gptneox_token_data_array_p, c_int, c_size_t]
_lib.gptneox_sample_top_k.restype = None

def gptneox_sample_top_p(ctx: gptneox_context_p, candidates, p: c_float, min_keep: c_size_t):
    if False:
        return 10
    return _lib.gptneox_sample_top_p(ctx, candidates, p, min_keep)
_lib.gptneox_sample_top_p.argtypes = [gptneox_context_p, gptneox_token_data_array_p, c_float, c_size_t]
_lib.gptneox_sample_top_p.restype = None

def gptneox_sample_tail_free(ctx: gptneox_context_p, candidates, z: c_float, min_keep: c_size_t):
    if False:
        for i in range(10):
            print('nop')
    return _lib.gptneox_sample_tail_free(ctx, candidates, z, min_keep)
_lib.gptneox_sample_tail_free.argtypes = [gptneox_context_p, gptneox_token_data_array_p, c_float, c_size_t]
_lib.gptneox_sample_tail_free.restype = None

def gptneox_sample_typical(ctx: gptneox_context_p, candidates, p: c_float, min_keep: c_size_t):
    if False:
        print('Hello World!')
    return _lib.gptneox_sample_typical(ctx, candidates, p, min_keep)
_lib.gptneox_sample_typical.argtypes = [gptneox_context_p, gptneox_token_data_array_p, c_float, c_size_t]
_lib.gptneox_sample_typical.restype = None

def gptneox_sample_temperature(ctx: gptneox_context_p, candidates, temp: c_float):
    if False:
        return 10
    return _lib.gptneox_sample_temperature(ctx, candidates, temp)
_lib.gptneox_sample_temperature.argtypes = [gptneox_context_p, gptneox_token_data_array_p, c_float]
_lib.gptneox_sample_temperature.restype = None

def gptneox_sample_token_mirostat(ctx: gptneox_context_p, candidates, tau: c_float, eta: c_float, m: c_int, mu) -> gptneox_token:
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)
_lib.gptneox_sample_token_mirostat.argtypes = [gptneox_context_p, gptneox_token_data_array_p, c_float, c_float, c_int, c_float_p]
_lib.gptneox_sample_token_mirostat.restype = gptneox_token

def gptneox_sample_token_mirostat_v2(ctx: gptneox_context_p, candidates, tau: c_float, eta: c_float, mu) -> gptneox_token:
    if False:
        print('Hello World!')
    return _lib.gptneox_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)
_lib.gptneox_sample_token_mirostat_v2.argtypes = [gptneox_context_p, gptneox_token_data_array_p, c_float, c_float, c_float_p]
_lib.gptneox_sample_token_mirostat_v2.restype = gptneox_token

def gptneox_sample_token_greedy(ctx: gptneox_context_p, candidates) -> gptneox_token:
    if False:
        i = 10
        return i + 15
    return _lib.gptneox_sample_token_greedy(ctx, candidates)
_lib.gptneox_sample_token_greedy.argtypes = [gptneox_context_p, gptneox_token_data_array_p]
_lib.gptneox_sample_token_greedy.restype = gptneox_token

def gptneox_sample_token(ctx: gptneox_context_p, candidates) -> gptneox_token:
    if False:
        print('Hello World!')
    return _lib.gptneox_sample_token(ctx, candidates)
_lib.gptneox_sample_token.argtypes = [gptneox_context_p, gptneox_token_data_array_p]
_lib.gptneox_sample_token.restype = gptneox_token

def gptneox_print_timings(ctx: gptneox_context_p):
    if False:
        i = 10
        return i + 15
    _lib.gptneox_print_timings(ctx)
_lib.gptneox_print_timings.argtypes = [gptneox_context_p]
_lib.gptneox_print_timings.restype = None

def gptneox_reset_timings(ctx: gptneox_context_p):
    if False:
        return 10
    _lib.gptneox_reset_timings(ctx)
_lib.gptneox_reset_timings.argtypes = [gptneox_context_p]
_lib.gptneox_reset_timings.restype = None

def gptneox_print_system_info() -> bytes:
    if False:
        return 10
    return _lib.gptneox_print_system_info()
_lib.gptneox_print_system_info.argtypes = []
_lib.gptneox_print_system_info.restype = c_char_p