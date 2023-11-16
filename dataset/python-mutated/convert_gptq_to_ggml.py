import os
import re
import sys
import json
import warnings
import struct
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor
from pathlib import Path
from bigdl.llm.utils.common.log4Error import invalidInputError

def write_header(fout, shape, dst_name, ftype_cur):
    if False:
        return 10
    sname = dst_name.encode('utf-8')
    fout.write(struct.pack('iii', len(shape), len(sname), ftype_cur))
    fout.write(struct.pack('i' * len(shape), *shape[::-1]))
    fout.write(sname)
    fout.seek(fout.tell() + 31 & -32)

def convert_non_q4(src_name, dst_name, model, fout):
    if False:
        i = 10
        return i + 15
    v = model[src_name]
    shape = v.shape
    print('Processing non-Q4 variable: ' + src_name + ' with shape: ', shape, ' and type: ', v.dtype)
    if len(shape) == 1:
        print('  Converting to float32')
        v = v.to(torch.float32)
    ftype_cur = {torch.float16: 1, torch.float32: 0}[v.dtype]
    write_header(fout, shape, dst_name, ftype_cur)
    v.numpy().tofile(fout)

def expandToInt4(qweight):
    if False:
        print('Hello World!')
    eweight = qweight.repeat(8, axis=2)
    eweight = eweight.astype(np.uint32)
    for i in range(0, eweight.shape[2]):
        offset = i % (32 // 4) * 4
        eweight[:, :, i] = eweight[:, :, i] >> offset & 2 ** 4 - 1
    return eweight

def to_ggml_int16(eweight):
    if False:
        for i in range(10):
            print('nop')
    qweight = np.zeros((eweight.shape[0], eweight.shape[1], eweight.shape[2] // 4), dtype=np.uint16)
    eweight = np.asarray(eweight, dtype=np.uint16)
    for i in range(0, qweight.shape[2]):
        qweight[:, :, i] = eweight[:, :, i * 2 + 0]
        qweight[:, :, i] |= eweight[:, :, i * 2 + 32] << 1 * 4
        qweight[:, :, i] |= eweight[:, :, i * 2 + 1] << 2 * 4
        qweight[:, :, i] |= eweight[:, :, i * 2 + 33] << 3 * 4
    return qweight.astype(np.int16)

def qzeros_to_zeros(qzeros, bits=4):
    if False:
        i = 10
        return i + 15
    zeros = np.zeros((qzeros.shape[0], qzeros.shape[1] * (32 // bits)), dtype=np.float32)
    i = 0
    col = 0
    while col < qzeros.shape[1]:
        for j in range(i, i + 32 // bits):
            zeros[:, j] = (qzeros[:, col] >> bits * (j - i) & 2 ** bits - 1) + 1
        i += 32 // bits
        col += 1
    return zeros

def convert_q4(src_name, dst_name, model, fout, n_head, permute=False):
    if False:
        i = 10
        return i + 15
    qzeros = model[f'{src_name}.qzeros'].numpy()
    zeros = qzeros_to_zeros(qzeros).T
    scales = model[f'{src_name}.scales'].numpy().T
    g_idx = model[f'{src_name}.g_idx'].numpy()
    qweight = model[f'{src_name}.qweight'].numpy().T
    invalidInputError(np.all(g_idx[:-1] <= g_idx[1:]), 'Act-order is not supported, please use a no act-order model.')
    ftype = 3
    shape = (qweight.shape[0], qweight.shape[1] * 8)
    print('Processing Q4 variable: ' + src_name + ' with shape: ', shape)
    addends = -zeros * scales
    addends_view = np.asarray(addends, dtype=np.float16).view(dtype=np.int16)
    scales_view = np.asarray(scales, dtype=np.float16).view(dtype=np.int16)
    expanded = expandToInt4(qweight.reshape([qweight.shape[0], qweight.shape[1] // 8, 8]))
    grouped = to_ggml_int16(expanded)
    if addends_view.shape[1] == grouped.shape[1]:
        addends_rep = np.atleast_3d(addends_view)
        scales_rep = np.atleast_3d(scales_view)
    else:
        addends_rep = np.atleast_3d(addends_view).repeat(grouped.shape[1] // addends_view.shape[1], axis=1)
        scales_rep = np.atleast_3d(scales_view).repeat(grouped.shape[1] // scales_view.shape[1], axis=1)
    blob = np.concatenate([scales_rep, addends_rep, grouped], axis=2, casting='no')
    if permute:
        blob = blob.reshape(n_head, 2, shape[0] // n_head // 2, *blob.shape[1:]).swapaxes(1, 2).reshape(blob.shape)
    write_header(fout, shape, dst_name, ftype)
    blob.tofile(fout)

def find_quantized_model_file(model_path):
    if False:
        return 10
    model_path = Path(model_path)
    for ext in ['.safetensors', '.pt']:
        found = list(model_path.glob(f'*{ext}'))
        if len(found) > 0:
            if len(found) != 1:
                warnings.warn(f'Detected {len(found)} {ext} model, use the first one {found[0]}.')
            print(f'Detected model file {found[0]}')
            return str(found[0])

def convert_gptq2ggml(model_path, output_path, tokenizer_path=None):
    if False:
        i = 10
        return i + 15
    input_path = find_quantized_model_file(model_path)
    if input_path.endswith('pt'):
        model = torch.load(input_path, map_location='cpu')
    elif input_path.endswith('safetensors'):
        from safetensors.torch import load_file
        model = load_file(input_path)
    else:
        invalidInputError(False, 'unknown input model path, only support .safetensors or .pt file.')
    (n_vocab, n_embd) = model['model.embed_tokens.weight'].shape
    layer_re = 'model\\.layers\\.([0-9]+)'
    n_layer = 1 + max((int(re.match(layer_re, name).group(1)) for name in model if re.match(layer_re, name)))
    n_mult = 256
    n_head = {32: 32, 40: 40, 60: 52, 80: 64}[n_layer]
    if not tokenizer_path:
        tokenizer_path = os.path.join(model_path, 'tokenizer.model')
        invalidInputError(os.path.isfile(tokenizer_path), f'tokenizer.model was not found under {tokenizer_path}.Please specify the tokenizer-path')
    tokenizer = SentencePieceProcessor(tokenizer_path)
    vocab_size = tokenizer.vocab_size()
    invalidInputError(vocab_size <= n_vocab, 'vocab size not match.')
    fout = open(output_path, 'wb')
    fout.write(b'ggjt'[::-1])
    values = [3, n_vocab, n_embd, n_mult, n_head, n_layer, n_embd // n_head, 4]
    fout.write(struct.pack('i' * len(values), *values))
    for i in range(vocab_size):
        if tokenizer.is_unknown(i):
            text = ' ⁇ '.encode('utf-8')
        elif tokenizer.is_control(i):
            text = b''
        elif tokenizer.is_byte(i):
            piece = tokenizer.id_to_piece(i)
            if len(piece) != 6:
                print(f'Invalid token: {piece}')
                sys.exit(1)
            byte_value = int(piece[3:-1], 16)
            text = struct.pack('B', byte_value)
        else:
            text = tokenizer.id_to_piece(i).replace('▁', ' ').encode('utf-8')
        fout.write(struct.pack('i', len(text)))
        fout.write(text)
        fout.write(struct.pack('f', tokenizer.get_score(i)))
    convert_non_q4('model.embed_tokens.weight', 'tok_embeddings.weight', model, fout)
    convert_non_q4('model.norm.weight', 'norm.weight', model, fout)
    convert_non_q4('lm_head.weight', 'output.weight', model, fout)
    for i in range(n_layer):
        convert_q4(f'model.layers.{i}.self_attn.q_proj', f'layers.{i}.attention.wq.weight', model, fout, n_head, permute=True)
        convert_q4(f'model.layers.{i}.self_attn.k_proj', f'layers.{i}.attention.wk.weight', model, fout, n_head, permute=True)
        convert_q4(f'model.layers.{i}.self_attn.v_proj', f'layers.{i}.attention.wv.weight', model, fout, n_head)
        convert_q4(f'model.layers.{i}.self_attn.o_proj', f'layers.{i}.attention.wo.weight', model, fout, n_head)
        convert_q4(f'model.layers.{i}.mlp.gate_proj', f'layers.{i}.feed_forward.w1.weight', model, fout, n_head)
        convert_q4(f'model.layers.{i}.mlp.down_proj', f'layers.{i}.feed_forward.w2.weight', model, fout, n_head)
        convert_q4(f'model.layers.{i}.mlp.up_proj', f'layers.{i}.feed_forward.w3.weight', model, fout, n_head)
        convert_non_q4(f'model.layers.{i}.input_layernorm.weight', f'layers.{i}.attention_norm.weight', model, fout)
        convert_non_q4(f'model.layers.{i}.post_attention_layernorm.weight', f'layers.{i}.ffn_norm.weight', model, fout)
    fout.close()
    print('Done. Output file: ' + output_path)
    print('')
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: convert-gptq-to-ggml.py llamaXXb-4bit.pt tokenizer.model out.bin\n')
        sys.exit(1)
    fname_model = sys.argv[1]
    fname_tokenizer = sys.argv[2]
    out_path = sys.argv[3]
    convert_gptq2ggml(fname_model, out_path, tokenizer_path=fname_tokenizer)