import sys
import os
import ctypes
from ctypes import c_int, c_float, c_char_p, c_void_p, c_bool, pointer, POINTER, _Pointer, Structure, Array, c_uint8, c_size_t
import pathlib
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.utils.utils import get_shared_lib_info

def _load_shared_library(lib_base_name: str):
    if False:
        i = 10
        return i + 15
    (_base_path, _lib_paths) = get_shared_lib_info(lib_base_name=lib_base_name)
    if 'LLAMA_CPP_LIB' in os.environ:
        lib_base_name = os.environ['LLAMA_CPP_LIB']
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]
    cdll_args = dict()
    if sys.platform == 'win32' and sys.version_info >= (3, 8):
        os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
        os.add_dll_directory(str(_base_path))
        os.environ['PATH'] = str(_base_path) + ';' + os.environ['PATH']
        if 'CUDA_PATH' in os.environ:
            os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))
            os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'lib'))
        cdll_args['winmode'] = 0
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)
            except Exception as e:
                invalidInputError(False, f"Failed to load shared library '{_lib_path}': {e}.")
    invalidInputError(False, f"Shared library with base name '{lib_base_name}' not found.")
_lib_base_name = 'llama'
_lib = _load_shared_library(_lib_base_name)
c_float_p = POINTER(c_float)
c_uint8_p = POINTER(c_uint8)
c_size_t_p = POINTER(c_size_t)
LLAMA_FILE_MAGIC_GGJT = ctypes.c_uint(1734830708)
LLAMA_FILE_MAGIC_GGLA = ctypes.c_uint(1734831201)
LLAMA_FILE_MAGIC_GGMF = ctypes.c_uint(1734831462)
LLAMA_FILE_MAGIC_GGML = ctypes.c_uint(1734831468)
LLAMA_FILE_MAGIC_GGSN = ctypes.c_uint(1734833006)
LLAMA_FILE_VERSION = c_int(3)
LLAMA_FILE_MAGIC = LLAMA_FILE_MAGIC_GGJT
LLAMA_FILE_MAGIC_UNVERSIONED = LLAMA_FILE_MAGIC_GGML
LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN
LLAMA_SESSION_VERSION = c_int(1)
llama_context_p = c_void_p
llama_token = c_int
llama_token_p = POINTER(llama_token)

class llama_token_data(Structure):
    _fields_ = [('id', llama_token), ('logit', c_float), ('p', c_float)]
llama_token_data_p = POINTER(llama_token_data)

class llama_token_data_array(Structure):
    _fields_ = [('data', llama_token_data_p), ('size', c_size_t), ('sorted', c_bool)]
llama_token_data_array_p = POINTER(llama_token_data_array)
llama_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)

class llama_context_params(Structure):
    _fields_ = [('n_ctx', c_int), ('n_gpu_layers', c_int), ('seed', c_int), ('f16_kv', c_bool), ('logits_all', c_bool), ('vocab_only', c_bool), ('use_mmap', c_bool), ('use_mlock', c_bool), ('embedding', c_bool), ('progress_callback', llama_progress_callback), ('progress_callback_user_data', c_void_p)]
llama_context_params_p = POINTER(llama_context_params)
LLAMA_FTYPE_ALL_F32 = c_int(0)
LLAMA_FTYPE_MOSTLY_F16 = c_int(1)
LLAMA_FTYPE_MOSTLY_Q4_0 = c_int(2)
LLAMA_FTYPE_MOSTLY_Q4_1 = c_int(3)
LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = c_int(4)
LLAMA_FTYPE_MOSTLY_Q8_0 = c_int(7)
LLAMA_FTYPE_MOSTLY_Q5_0 = c_int(8)
LLAMA_FTYPE_MOSTLY_Q5_1 = c_int(9)

def llama_context_default_params() -> llama_context_params:
    if False:
        for i in range(10):
            print('nop')
    return _lib.llama_context_default_params()
_lib.llama_context_default_params.argtypes = []
_lib.llama_context_default_params.restype = llama_context_params

def llama_mmap_supported() -> bool:
    if False:
        i = 10
        return i + 15
    return _lib.llama_mmap_supported()
_lib.llama_mmap_supported.argtypes = []
_lib.llama_mmap_supported.restype = c_bool

def llama_mlock_supported() -> bool:
    if False:
        print('Hello World!')
    return _lib.llama_mlock_supported()
_lib.llama_mlock_supported.argtypes = []
_lib.llama_mlock_supported.restype = c_bool

def llama_init_backend():
    if False:
        i = 10
        return i + 15
    return _lib.llama_init_backend()
_lib.llama_init_backend.argtypes = []
_lib.llama_init_backend.restype = None

def llama_time_us() -> int:
    if False:
        i = 10
        return i + 15
    return _lib.llama_time_us()
_lib.llama_time_us.argtypes = []
_lib.llama_time_us.restype = ctypes.c_int64

def llama_init_from_file(path_model: bytes, params: llama_context_params) -> llama_context_p:
    if False:
        for i in range(10):
            print('nop')
    return _lib.llama_init_from_file(path_model, pointer(params))
_lib.llama_init_from_file.argtypes = [c_char_p, llama_context_params_p]
_lib.llama_init_from_file.restype = llama_context_p

def llama_free(ctx: llama_context_p):
    if False:
        while True:
            i = 10
    return _lib.llama_free(ctx)
_lib.llama_free.argtypes = [llama_context_p]
_lib.llama_free.restype = None

def llama_model_quantize(fname_inp: bytes, fname_out: bytes, ftype: c_int, nthread: c_int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return _lib.llama_model_quantize(fname_inp, fname_out, ftype, nthread)
_lib.llama_model_quantize.argtypes = [c_char_p, c_char_p, c_int, c_int]
_lib.llama_model_quantize.restype = c_int

def llama_apply_lora_from_file(ctx: llama_context_p, path_lora: c_char_p, path_base_model: c_char_p, n_threads: c_int) -> int:
    if False:
        while True:
            i = 10
    return _lib.llama_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads)
_lib.llama_apply_lora_from_file.argtypes = [llama_context_p, c_char_p, c_char_p, c_int]
_lib.llama_apply_lora_from_file.restype = c_int

def llama_get_kv_cache_token_count(ctx: llama_context_p) -> int:
    if False:
        print('Hello World!')
    return _lib.llama_get_kv_cache_token_count(ctx)
_lib.llama_get_kv_cache_token_count.argtypes = [llama_context_p]
_lib.llama_get_kv_cache_token_count.restype = c_int

def llama_set_rng_seed(ctx: llama_context_p, seed: c_int):
    if False:
        return 10
    return _lib.llama_set_rng_seed(ctx, seed)
_lib.llama_set_rng_seed.argtypes = [llama_context_p, c_int]
_lib.llama_set_rng_seed.restype = None

def llama_get_state_size(ctx: llama_context_p) -> int:
    if False:
        i = 10
        return i + 15
    return _lib.llama_get_state_size(ctx)
_lib.llama_get_state_size.argtypes = [llama_context_p]
_lib.llama_get_state_size.restype = c_size_t

def llama_copy_state_data(ctx: llama_context_p, dst) -> int:
    if False:
        for i in range(10):
            print('nop')
    return _lib.llama_copy_state_data(ctx, dst)
_lib.llama_copy_state_data.argtypes = [llama_context_p, c_uint8_p]
_lib.llama_copy_state_data.restype = c_size_t

def llama_set_state_data(ctx: llama_context_p, src) -> int:
    if False:
        while True:
            i = 10
    return _lib.llama_set_state_data(ctx, src)
_lib.llama_set_state_data.argtypes = [llama_context_p, c_uint8_p]
_lib.llama_set_state_data.restype = c_size_t

def llama_load_session_file(ctx: llama_context_p, path_session: bytes, tokens_out, n_token_capacity: c_size_t, n_token_count_out) -> int:
    if False:
        print('Hello World!')
    return _lib.llama_load_session_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)
_lib.llama_load_session_file.argtypes = [llama_context_p, c_char_p, llama_token_p, c_size_t, c_size_t_p]
_lib.llama_load_session_file.restype = c_size_t

def llama_save_session_file(ctx: llama_context_p, path_session: bytes, tokens, n_token_count: c_size_t) -> int:
    if False:
        return 10
    return _lib.llama_save_session_file(ctx, path_session, tokens, n_token_count)
_lib.llama_save_session_file.argtypes = [llama_context_p, c_char_p, llama_token_p, c_size_t]
_lib.llama_save_session_file.restype = c_size_t

def llama_eval(ctx: llama_context_p, tokens, n_tokens: c_int, n_past: c_int, n_threads: c_int) -> int:
    if False:
        return 10
    return _lib.llama_eval(ctx, tokens, n_tokens, n_past, n_threads)
_lib.llama_eval.argtypes = [llama_context_p, llama_token_p, c_int, c_int, c_int]
_lib.llama_eval.restype = c_int

def llama_tokenize(ctx: llama_context_p, text: bytes, tokens, n_max_tokens: c_int, add_bos: c_bool) -> int:
    if False:
        for i in range(10):
            print('nop')
    return _lib.llama_tokenize(ctx, text, tokens, n_max_tokens, add_bos)
_lib.llama_tokenize.argtypes = [llama_context_p, c_char_p, llama_token_p, c_int, c_bool]
_lib.llama_tokenize.restype = c_int

def llama_n_vocab(ctx: llama_context_p) -> int:
    if False:
        for i in range(10):
            print('nop')
    return _lib.llama_n_vocab(ctx)
_lib.llama_n_vocab.argtypes = [llama_context_p]
_lib.llama_n_vocab.restype = c_int

def llama_n_ctx(ctx: llama_context_p) -> int:
    if False:
        for i in range(10):
            print('nop')
    return _lib.llama_n_ctx(ctx)
_lib.llama_n_ctx.argtypes = [llama_context_p]
_lib.llama_n_ctx.restype = c_int

def llama_n_embd(ctx: llama_context_p) -> int:
    if False:
        print('Hello World!')
    return _lib.llama_n_embd(ctx)
_lib.llama_n_embd.argtypes = [llama_context_p]
_lib.llama_n_embd.restype = c_int

def llama_get_logits(ctx: llama_context_p):
    if False:
        return 10
    return _lib.llama_get_logits(ctx)
_lib.llama_get_logits.argtypes = [llama_context_p]
_lib.llama_get_logits.restype = c_float_p

def llama_get_embeddings(ctx: llama_context_p):
    if False:
        for i in range(10):
            print('nop')
    return _lib.llama_get_embeddings(ctx)
_lib.llama_get_embeddings.argtypes = [llama_context_p]
_lib.llama_get_embeddings.restype = c_float_p

def llama_token_to_str(ctx: llama_context_p, token: llama_token) -> bytes:
    if False:
        while True:
            i = 10
    return _lib.llama_token_to_str(ctx, token)
_lib.llama_token_to_str.argtypes = [llama_context_p, llama_token]
_lib.llama_token_to_str.restype = c_char_p

def llama_token_bos() -> int:
    if False:
        return 10
    return _lib.llama_token_bos()
_lib.llama_token_bos.argtypes = []
_lib.llama_token_bos.restype = llama_token

def llama_token_eos() -> int:
    if False:
        for i in range(10):
            print('nop')
    return _lib.llama_token_eos()
_lib.llama_token_eos.argtypes = []
_lib.llama_token_eos.restype = llama_token

def llama_token_nl() -> int:
    if False:
        for i in range(10):
            print('nop')
    return _lib.llama_token_nl()
_lib.llama_token_nl.argtypes = []
_lib.llama_token_nl.restype = llama_token

def llama_init_candidates(ctx: llama_context_p, candidates):
    if False:
        print('Hello World!')
    return _lib.llama_init_candidates(ctx, candidates)
_lib.llama_init_candidates.argtypes = [llama_context_p, llama_token_data_array_p]
_lib.llama_init_candidates.restype = None

def llama_sample_repetition_penalty(ctx: llama_context_p, candidates, last_tokens_data, last_tokens_size: c_int, penalty: c_float):
    if False:
        while True:
            i = 10
    return _lib.llama_sample_repetition_penalty(ctx, candidates, last_tokens_data, last_tokens_size, penalty)
_lib.llama_sample_repetition_penalty.argtypes = [llama_context_p, llama_token_data_array_p, llama_token_p, c_int, c_float]
_lib.llama_sample_repetition_penalty.restype = None

def llama_sample_frequency_and_presence_penalties(ctx: llama_context_p, candidates, last_tokens_data, last_tokens_size: c_int, alpha_frequency: c_float, alpha_presence: c_float):
    if False:
        print('Hello World!')
    return _lib.llama_sample_frequency_and_presence_penalties(ctx, candidates, last_tokens_data, last_tokens_size, alpha_frequency, alpha_presence)
_lib.llama_sample_frequency_and_presence_penalties.argtypes = [llama_context_p, llama_token_data_array_p, llama_token_p, c_int, c_float, c_float]
_lib.llama_sample_frequency_and_presence_penalties.restype = None

def llama_sample_softmax(ctx: llama_context_p, candidates):
    if False:
        print('Hello World!')
    return _lib.llama_sample_softmax(ctx, candidates)
_lib.llama_sample_softmax.argtypes = [llama_context_p, llama_token_data_array_p]
_lib.llama_sample_softmax.restype = None

def llama_sample_top_k(ctx: llama_context_p, candidates, k: c_int, min_keep: c_size_t):
    if False:
        while True:
            i = 10
    return _lib.llama_sample_top_k(ctx, candidates, k, min_keep)
_lib.llama_sample_top_k.argtypes = [llama_context_p, llama_token_data_array_p, c_int, c_size_t]
_lib.llama_sample_top_k.restype = None

def llama_sample_top_p(ctx: llama_context_p, candidates, p: c_float, min_keep: c_size_t):
    if False:
        while True:
            i = 10
    return _lib.llama_sample_top_p(ctx, candidates, p, min_keep)
_lib.llama_sample_top_p.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_size_t]
_lib.llama_sample_top_p.restype = None

def llama_sample_tail_free(ctx: llama_context_p, candidates, z: c_float, min_keep: c_size_t):
    if False:
        i = 10
        return i + 15
    return _lib.llama_sample_tail_free(ctx, candidates, z, min_keep)
_lib.llama_sample_tail_free.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_size_t]
_lib.llama_sample_tail_free.restype = None

def llama_sample_typical(ctx: llama_context_p, candidates, p: c_float, min_keep: c_size_t):
    if False:
        for i in range(10):
            print('nop')
    return _lib.llama_sample_typical(ctx, candidates, p, min_keep)
_lib.llama_sample_typical.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_size_t]
_lib.llama_sample_typical.restype = None

def llama_sample_temperature(ctx: llama_context_p, candidates, temp: c_float):
    if False:
        while True:
            i = 10
    return _lib.llama_sample_temperature(ctx, candidates, temp)
_lib.llama_sample_temperature.argtypes = [llama_context_p, llama_token_data_array_p, c_float]
_lib.llama_sample_temperature.restype = None

def llama_sample_token_mirostat(ctx: llama_context_p, candidates, tau: c_float, eta: c_float, m: c_int, mu) -> int:
    if False:
        return 10
    return _lib.llama_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)
_lib.llama_sample_token_mirostat.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_float, c_int, c_float_p]
_lib.llama_sample_token_mirostat.restype = llama_token

def llama_sample_token_mirostat_v2(ctx: llama_context_p, candidates, tau: c_float, eta: c_float, mu) -> int:
    if False:
        return 10
    return _lib.llama_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)
_lib.llama_sample_token_mirostat_v2.argtypes = [llama_context_p, llama_token_data_array_p, c_float, c_float, c_float_p]
_lib.llama_sample_token_mirostat_v2.restype = llama_token

def llama_sample_token_greedy(ctx: llama_context_p, candidates) -> int:
    if False:
        print('Hello World!')
    return _lib.llama_sample_token_greedy(ctx, candidates)
_lib.llama_sample_token_greedy.argtypes = [llama_context_p, llama_token_data_array_p]
_lib.llama_sample_token_greedy.restype = llama_token

def llama_sample_token(ctx: llama_context_p, candidates) -> int:
    if False:
        i = 10
        return i + 15
    return _lib.llama_sample_token(ctx, candidates)
_lib.llama_sample_token.argtypes = [llama_context_p, llama_token_data_array_p]
_lib.llama_sample_token.restype = llama_token

def llama_print_timings(ctx: llama_context_p):
    if False:
        while True:
            i = 10
    _lib.llama_print_timings(ctx)
_lib.llama_print_timings.argtypes = [llama_context_p]
_lib.llama_print_timings.restype = None

def llama_reset_timings(ctx: llama_context_p):
    if False:
        print('Hello World!')
    _lib.llama_reset_timings(ctx)
_lib.llama_reset_timings.argtypes = [llama_context_p]
_lib.llama_reset_timings.restype = None

def llama_print_system_info() -> bytes:
    if False:
        print('Hello World!')
    return _lib.llama_print_system_info()
_lib.llama_print_system_info.argtypes = []
_lib.llama_print_system_info.restype = c_char_p

def ggml_quantize_tensor(src, dst: ctypes.c_void_p, qtype: ctypes.c_int, n: ctypes.c_size_t, k: ctypes.c_int, hist) -> int:
    if False:
        i = 10
        return i + 15
    return _lib.ggml_quantize_tensor(src, dst, qtype, n, k, hist)
_lib.ggml_quantize_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
_lib.ggml_quantize_tensor.restype = ctypes.c_size_t

def ggml_type_size(qtype: ctypes.c_int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return _lib.ggml_type_size(qtype)
_lib.ggml_type_size.argtypes = [ctypes.c_int]
_lib.ggml_type_size.restype = ctypes.c_size_t

def ggml_qk_size(qtype: ctypes.c_int) -> int:
    if False:
        print('Hello World!')
    return _lib.ggml_qk_size(qtype)
_lib.ggml_qk_size.argtypes = [ctypes.c_int]
_lib.ggml_qk_size.restype = ctypes.c_int

def ggml_dequantize_q4_0(src: ctypes.c_void_p, dst: ctypes.c_void_p, k: ctypes.c_size_t):
    if False:
        i = 10
        return i + 15
    _lib.ggml_dequantize_q4_0(src, dst, k)
_lib.ggml_dequantize_q4_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_lib.ggml_quantize_q4_0.restype = None

def ggml_dequantize(src: ctypes.c_void_p, dst: ctypes.c_void_p, k: ctypes.c_size_t, qtype: ctypes.c_int):
    if False:
        i = 10
        return i + 15
    _lib.ggml_dequantize(src, dst, k, qtype)
_lib.ggml_dequantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.ggml_dequantize.restype = None

def ggml_q_format_convet_cpu2xpu(src: ctypes.c_void_p, dst: ctypes.c_void_p, n: ctypes.c_size_t, qtype: ctypes.c_int):
    if False:
        return 10
    _lib.ggml_q_format_convet_cpu2xpu(src, dst, n, qtype)
_lib.ggml_q_format_convet_cpu2xpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.ggml_q_format_convet_cpu2xpu.restype = None

def ggml_q_format_convet_xpu2cpu(src: ctypes.c_void_p, dst: ctypes.c_void_p, n: ctypes.c_size_t, qtype: ctypes.c_int):
    if False:
        i = 10
        return i + 15
    _lib.ggml_q_format_convet_xpu2cpu(src, dst, n, qtype)
_lib.ggml_q_format_convet_xpu2cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.ggml_q_format_convet_xpu2cpu.restype = None

def ggml_compute_forward_mul_mat_q_fp32(src_0_ne, src_0_data, src_0_qtype, src_1_ne, src_1_data, result) -> None:
    if False:
        print('Hello World!')
    return _lib.ggml_compute_forward_mul_mat_q_fp32(src_0_ne, src_0_data, src_0_qtype, src_1_ne, src_1_data, result)
_lib.ggml_compute_forward_mul_mat_q_fp32.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_void_p, ctypes.c_void_p]
_lib.ggml_compute_forward_mul_mat_q_fp32.restype = None
_llama_initialized = False
if not _llama_initialized:
    llama_init_backend()
    _llama_initialized = True