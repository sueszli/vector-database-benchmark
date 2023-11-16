import argparse
import collections
import os
import pickle
import re
import numpy as np

def define_template(**kwargs):
    if False:
        return 10
    template = '\n    float cuda{cuda_arch}_{conv_type}_time_pred[{out_dim}] = {{0.0f}};\n    float cuda{cuda_arch}_{conv_type}_mask[{out_dim}] = {{0.0f}};\n    float cuda{cuda_arch}_{conv_type}_hidden_units[{hidden_num}] = {{0.0f}};\n    const static size_t cuda{cuda_arch}_{conv_type}_layers_dim[{layer_num}] = {{{layers_dim}}};\n    const static float cuda{cuda_arch}_{conv_type}_matrices[{matrices_dim}] = {{{matrices}}};\n    const static float cuda{cuda_arch}_{conv_type}_biases[{biases_dim}] = {{{biases}}};\n    const static float cuda{cuda_arch}_{conv_type}_alpha[{out_dim}] = {{{alpha}}};\n    const static float cuda{cuda_arch}_{conv_type}_beta[{out_dim}] = {{{beta}}};\n    '
    return template.format(**kwargs)

def cudnn_slt_template(**kwargs):
    if False:
        return 10
    template = '#if CUDNN_MAJOR == {cudnn_major} && CUDNN_MINOR == {cudnn_minor}\n' + '    {define_cmd}\n' + '    {select_cmd}\n' + '    return true;\n' + '#endif\n'
    return template.format(**kwargs)

def select_template(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    template = 'if (conv_type == ConvolutionType::{conv_type} && cuda_major == {cuda_major} &&\n               cuda_minor == {cuda_minor}) {{\n        *layer_num_p = {layer_num};\n        *hidden_units_p = cuda{cuda_arch}_{conv_type}_hidden_units;\n        *layers_dim_p = cuda{cuda_arch}_{conv_type}_layers_dim;\n        *matrices_p = cuda{cuda_arch}_{conv_type}_matrices;\n        *biases_p = cuda{cuda_arch}_{conv_type}_biases;\n        *alpha_p = cuda{cuda_arch}_{conv_type}_alpha;\n        *beta_p = cuda{cuda_arch}_{conv_type}_beta;\n        *time_pred_p = cuda{cuda_arch}_{conv_type}_time_pred;\n        *mask_p = cuda{cuda_arch}_{conv_type}_mask;\n    }} else '
    return template.format(**kwargs)

def main():
    if False:
        return 10
    fill_src()

def fill_src():
    if False:
        while True:
            i = 10
    home = os.path.dirname(__file__)
    matrix_files = os.listdir(os.path.join(home, 'params'))
    gen_list = collections.defaultdict(list)
    cudnn_slt_cmd = ''
    if len(matrix_files) == 0:
        print('Warning: no param files detected.')
    for fpath in matrix_files:
        cudnn_version = re.findall('cudnn([\\d.]+)', fpath)[0]
        gen_list[cudnn_version].append(fpath)
    for cudnn in gen_list:
        select_cmd = '{\n' + ' ' * 8 + 'return false;\n' + ' ' * 4 + '}'
        define_cmd = ''
        (cudnn_major, cudnn_minor) = cudnn.split('.')
        for fpath in gen_list[cudnn]:
            cuda_arch = fpath.split('-')[1].replace('.', '_')
            print('cudnn_version: {}, cuda_arch: {}'.format(cudnn, cuda_arch))
            conv_type = fpath.split('-')[2].split('.')[0]
            with open(os.path.join(home, 'params/{}'.format(fpath)), 'rb') as pobj:
                params = pickle.load(pobj)
                (crt_define_cmd, crt_select_cmd) = gen_cmds(cuda_arch, conv_type, params)
                select_cmd = crt_select_cmd + select_cmd
                define_cmd = crt_define_cmd + define_cmd
        cudnn_slt_cmd += cudnn_slt_template(cudnn_major=cudnn_major, cudnn_minor=cudnn_minor, select_cmd=select_cmd, define_cmd=define_cmd)
    with open(os.path.join(home, 'get_params.template'), 'r') as srcf:
        src = srcf.read()
    dst = src.replace('{cudnn_select}', cudnn_slt_cmd)
    MegDNN_path = os.path.join(home, '../..')
    with open(os.path.join(MegDNN_path, 'src/cuda/convolution/get_params.cpp'), 'w') as dstf:
        dstf.write(dst)

def gen_cmds(cuda_arch, conv_type, params):
    if False:
        return 10
    (cuda_major, cuda_minor) = cuda_arch.split('_')
    alphastr = format_array(params['alpha']).rstrip()[:-1]
    betastr = format_array(params['beta']).rstrip()[:-1]
    W_list = params['W']
    b_list = params['b']
    Wstr = ''
    bstr = ''
    layer_num = str(len(b_list) + 1)
    layers_dim = [W_list[0].shape[1]]
    matrices_dim = 0
    biases_dim = 0
    for W in W_list:
        Wstr += format_array(W)
        matrices_dim += W.shape[0] * W.shape[1]
    for b in b_list:
        bstr += format_array(b)
        layers_dim.append(b.shape[0])
        biases_dim += b.shape[0]
    Wstr = Wstr.rstrip()[:-1]
    bstr = bstr.rstrip()[:-1]
    hidden_num = sum(layers_dim[1:-1])
    out_dim = layers_dim[-1]
    layers_dim_str = format_array(np.array(layers_dim)).rstrip()[:-1]
    select_cmd = select_template(conv_type=conv_type.upper(), cuda_major=cuda_major, cuda_minor=cuda_minor, layer_num=layer_num, cuda_arch=cuda_arch)
    define_cmd = define_template(cuda_arch=cuda_arch, conv_type=conv_type.upper(), hidden_num=hidden_num, layer_num=layer_num, out_dim=out_dim, layers_dim=layers_dim_str, matrices_dim=matrices_dim, matrices=Wstr, biases_dim=biases_dim, biases=bstr, alpha=alphastr, beta=betastr)
    return (define_cmd, select_cmd)

def format_array(array):
    if False:
        print('Hello World!')
    flat_array = np.squeeze(array.reshape(1, -1))
    array_str = ''
    ind = 0
    if flat_array.dtype == 'int':
        for ind in range(len(flat_array)):
            array_str += str(flat_array[ind]) + ', '
    else:
        for ind in range(len(flat_array)):
            if ind % 4 == 0:
                array_str += '\n' + ' ' * 12
            ele = flat_array[ind]
            if abs(ele) < 1e-37:
                array_str += '0.0, '
            else:
                array_str += '{:.6e}, '.format(ele)
    return array_str
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate cuDNN heuristic code by neural network into {MEGDNN_ROOT}/src/cuda/convolution/get_params.cpp, using parameter value from pickle files in {MEGDNN_ROOT}/scripts/gen_heuristic/params/')
    args = parser.parse_args()
    main()