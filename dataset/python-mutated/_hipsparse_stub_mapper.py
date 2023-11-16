import urllib.request
import sys
from typing import Tuple
cusparse_h = '/usr/local/cuda-{0}/include/cusparse.h'
cu_versions = ('11.3', '11.0', '10.2')
hipsparse_url = 'https://raw.githubusercontent.com/ROCmSoftwarePlatform/hipSPARSE/rocm-{0}/library/include/hipsparse.h'
hip_versions = ('3.5.0', '3.7.0', '3.8.0', '3.9.0', '4.0.0', '4.2.0')
typedefs: Tuple[str, ...]
typedefs = ('cusparseIndexBase_t', 'cusparseStatus_t', 'cusparseHandle_t', 'cusparseMatDescr_t', 'csrsv2Info_t', 'csrsm2Info_t', 'csric02Info_t', 'bsric02Info_t', 'csrilu02Info_t', 'bsrilu02Info_t', 'csrgemm2Info_t', 'cusparseMatrixType_t', 'cusparseFillMode_t', 'cusparseDiagType_t', 'cusparsePointerMode_t', 'cusparseAction_t', 'cusparseDirection_t', 'cusparseSolvePolicy_t', 'cusparseOperation_t')
typedefs += ('cusparseSpVecDescr_t', 'cusparseDnVecDescr_t', 'cusparseSpMatDescr_t', 'cusparseDnMatDescr_t', 'cusparseIndexType_t', 'cusparseFormat_t', 'cusparseOrder_t', 'cusparseSpMVAlg_t', 'cusparseSpMMAlg_t', 'cusparseSparseToDenseAlg_t', 'cusparseDenseToSparseAlg_t', 'cusparseCsr2CscAlg_t')
cudaDataType_converter = '\n#if HIP_VERSION >= 402\nstatic hipDataType convert_hipDatatype(cudaDataType type) {\n    switch(static_cast<int>(type)) {\n        case 2 /* CUDA_R_16F */: return HIP_R_16F;\n        case 0 /* CUDA_R_32F */: return HIP_R_32F;\n        case 1 /* CUDA_R_64F */: return HIP_R_64F;\n        case 6 /* CUDA_C_16F */: return HIP_C_16F;\n        case 4 /* CUDA_C_32F */: return HIP_C_32F;\n        case 5 /* CUDA_C_64F */: return HIP_C_64F;\n        default: throw std::runtime_error("unrecognized type");\n    }\n}\n#endif\n'
cusparseOrder_converter = '\n#if HIP_VERSION >= 402\ntypedef enum {} cusparseOrder_t;\nstatic hipsparseOrder_t convert_hipsparseOrder_t(cusparseOrder_t type) {\n    switch(static_cast<int>(type)) {\n        case 1 /* CUSPARSE_ORDER_COL */: return HIPSPARSE_ORDER_COLUMN;\n        case 2 /* CUSPARSE_ORDER_ROW */: return HIPSPARSE_ORDER_ROW;\n        default: throw std::runtime_error("unrecognized type");\n    }\n}\n'
default_return_code = '\n#if HIP_VERSION < 401\n#define HIPSPARSE_STATUS_NOT_SUPPORTED (hipsparseStatus_t)10\n#endif\n'
processed_typedefs = set()

def get_idx_to_func(cu_h, cu_func):
    if False:
        i = 10
        return i + 15
    cu_sig = cu_h.find(cu_func)
    while True:
        if cu_sig == -1:
            break
        elif cu_h[cu_sig + len(cu_func)] != '(':
            cu_sig = cu_h.find(cu_func, cu_sig + 1)
        else:
            break
    return cu_sig

def get_hip_ver_num(hip_version):
    if False:
        i = 10
        return i + 15
    hip_version = hip_version.split('.')
    return int(hip_version[0]) * 100 + int(hip_version[1])

def merge_bad_broken_lines(cu_sig):
    if False:
        while True:
            i = 10
    cu_sig_processed = []
    skip_line = None
    for (line, s) in enumerate(cu_sig):
        if line != skip_line:
            if s.endswith(',') or s.endswith(')'):
                cu_sig_processed.append(s)
            else:
                break_idx = s.find(',')
                if break_idx == -1:
                    break_idx = s.find(')')
                if break_idx == -1:
                    cu_sig_processed.append(s + cu_sig[line + 1])
                    skip_line = line + 1
                else:
                    cu_sig_processed.append(s[:break_idx + 1])
    return cu_sig_processed

def process_func_args(s, hip_sig, decl, hip_func):
    if False:
        i = 10
        return i + 15
    if 'const cuComplex*' in s:
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'reinterpret_cast<const hipComplex*>'
    elif 'const cuDoubleComplex*' in s:
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'reinterpret_cast<const hipDoubleComplex*>'
    elif 'cuComplex*' in s:
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'reinterpret_cast<hipComplex*>'
    elif 'cuDoubleComplex*' in s:
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'reinterpret_cast<hipDoubleComplex*>'
    elif 'cuComplex' in s:
        s = s.split()
        decl += '  hipComplex blah;\n'
        decl += f'  blah.x={s[-1][:-1]}.x;\n  blah.y={s[-1][:-1]}.y;\n'
        arg = 'blah' + s[-1][-1]
        cast = ''
    elif 'cuDoubleComplex' in s:
        s = s.split()
        decl += '  hipDoubleComplex blah;\n'
        decl += f'  blah.x={s[-1][:-1]}.x;\n  blah.y={s[-1][:-1]}.y;\n'
        arg = 'blah' + s[-1][-1]
        cast = ''
    elif 'cudaDataType*' in s:
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'reinterpret_cast<hipDataType*>'
    elif 'cudaDataType' in s:
        s = s.split()
        decl += '  hipDataType blah = convert_hipDatatype('
        decl += s[-1][:-1] + ');\n'
        arg = 'blah' + s[-1][-1]
        cast = ''
    elif 'cusparseOrder_t*' in s:
        s = s.split()
        decl += '  hipsparseOrder_t blah2 = '
        decl += 'convert_hipsparseOrder_t(*' + s[-1][:-1] + ');\n'
        arg = '&blah2' + s[-1][-1]
        cast = ''
    elif 'cusparseOrder_t' in s:
        s = s.split()
        decl += '  hipsparseOrder_t blah2 = '
        decl += 'convert_hipsparseOrder_t(' + s[-1][:-1] + ');\n'
        arg = 'blah2' + s[-1][-1]
        cast = ''
    elif 'const void*' in s and hip_func == 'hipsparseSpVV_bufferSize':
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'const_cast<void*>'
    else:
        s = s.split()
        arg = s[-1]
        cast = ''
    hip_sig += cast + arg + ' '
    return (hip_sig, decl)

def main(hip_h, cu_h, stubs, hip_version, init):
    if False:
        while True:
            i = 10
    hip_version = get_hip_ver_num(hip_version)
    hip_stub_h = []
    for (i, line) in enumerate(stubs):
        if i == 3 and (not init):
            hip_stub_h.append(line)
            if hip_version == 305:
                hip_stub_h.append('#include <hipsparse.h>')
                hip_stub_h.append('#include <hip/hip_version.h>    // for HIP_VERSION')
                hip_stub_h.append('#include <hip/library_types.h>  // for hipDataType')
                hip_stub_h.append(cudaDataType_converter)
                hip_stub_h.append(default_return_code)
        elif line.startswith('typedef'):
            old_line = ''
            typedef_found = False
            typedef_needed = True
            for t in typedefs:
                if t in line and t not in processed_typedefs:
                    hip_t = 'hip' + t[2:] if t.startswith('cu') else t
                    if hip_t in hip_h:
                        old_line = line
                        if t != hip_t:
                            old_line = line
                            line = 'typedef ' + hip_t + ' ' + t + ';'
                        else:
                            if hip_version == 305:
                                line = None
                            typedef_needed = False
                        typedef_found = True
                    else:
                        pass
                    break
            else:
                t = None
            if line is not None:
                if t == 'cusparseOrder_t' and hip_version == 402:
                    hip_stub_h.append(cusparseOrder_converter)
                elif typedef_found and hip_version > 305:
                    if typedef_needed:
                        hip_stub_h.append(f'#if HIP_VERSION >= {hip_version}')
                    else:
                        hip_stub_h.append(f'#if HIP_VERSION < {hip_version}')
                if not (t == 'cusparseOrder_t' and hip_version == 402):
                    hip_stub_h.append(line)
                if typedef_found and hip_version > 305:
                    if typedef_needed:
                        hip_stub_h.append('#else')
                        hip_stub_h.append(old_line)
                    hip_stub_h.append('#endif\n')
            if t is not None and typedef_found:
                processed_typedefs.add(t)
        elif '...' in line:
            sig = line.split()
            try:
                assert len(sig) == 3
            except AssertionError:
                print(f'sig is {sig}')
                raise
            cu_func = sig[1]
            cu_func = cu_func[:cu_func.find('(')]
            hip_func = 'hip' + cu_func[2:]
            cu_sig = get_idx_to_func(cu_h, cu_func)
            hip_sig = get_idx_to_func(hip_h, hip_func)
            if cu_sig == -1 and hip_sig == -1:
                assert False
            elif cu_sig == -1 and hip_sig != -1:
                print(cu_func, 'not found in cuSPARSE, maybe removed?', file=sys.stderr)
                can_map = False
            elif cu_sig != -1 and hip_sig == -1:
                print(hip_func, 'not found in hipSPARSE, maybe not supported?', file=sys.stderr)
                can_map = False
            else:
                end_idx = cu_h[cu_sig:].find(')')
                assert end_idx != -1
                cu_sig = cu_h[cu_sig:cu_sig + end_idx + 1]
                cu_sig = cu_sig.split('\n')
                new_cu_sig = cu_sig[0] + '\n'
                for s in cu_sig[1:]:
                    new_cu_sig += ' ' * (len(sig[0]) + 1) + s + '\n'
                cu_sig = new_cu_sig[:-1]
                sig[1] = cu_sig
                can_map = True
            hip_stub_h.append(' '.join(sig))
            line = stubs[i + 1]
            if 'return' not in line:
                line = stubs[i + 2]
                assert 'return' in line
            if can_map:
                cu_sig = cu_sig.split('\n')
                cu_sig = merge_bad_broken_lines(cu_sig)
                if hip_version != 305:
                    hip_stub_h.append(f'#if HIP_VERSION >= {hip_version}')
                hip_sig = '  return ' + hip_func + '('
                decl = ''
                for s in cu_sig:
                    (hip_sig, decl) = process_func_args(s, hip_sig, decl, hip_func)
                hip_sig = hip_sig[:-1] + ';'
                hip_stub_h.append(decl + hip_sig)
                if hip_version != 305:
                    hip_stub_h.append('#else')
                    hip_stub_h.append('  return HIPSPARSE_STATUS_NOT_SUPPORTED;')
                    hip_stub_h.append('#endif')
            else:
                hip_stub_h.append(line[:line.find('return') + 6] + ' HIPSPARSE_STATUS_NOT_SUPPORTED;')
        elif 'return' in line:
            if 'CUSPARSE_STATUS' in line:
                pass
            elif 'HIPSPARSE_STATUS_NOT_SUPPORTED' in line:
                if '#else' in stubs[i - 1]:
                    hip_stub_h.append(line)
            else:
                hip_stub_h.append(line)
        else:
            hip_stub_h.append(line)
    return '\n'.join(hip_stub_h) + '\n'
if __name__ == '__main__':
    with open('cupy_backends/stub/cupy_cusparse.h', 'r') as f:
        stubs = f.read()
    init = False
    for cu_ver in cu_versions:
        with open(cusparse_h.format(cu_ver), 'r') as f:
            cu_h = f.read()
        x = 0
        for hip_ver in hip_versions:
            stubs_splitted = stubs.splitlines()
            req = urllib.request.urlopen(hipsparse_url.format(hip_ver))
            with req as f:
                hip_h = f.read().decode()
            stubs = main(hip_h, cu_h, stubs_splitted, hip_ver, init)
            init = True
    stubs = stubs.replace('#define CUSPARSE_VERSION -1', '#define CUSPARSE_VERSION (hipsparseVersionMajor*100000+hipsparseVersionMinor*100+hipsparseVersionPatch)')
    stubs = stubs.replace('INCLUDE_GUARD_STUB_CUPY_CUSPARSE_H', 'INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H')
    stubs = stubs[stubs.find('\n'):]
    with open('cupy_backends/hip/cupy_hipsparse.h', 'w') as f:
        f.write(stubs)