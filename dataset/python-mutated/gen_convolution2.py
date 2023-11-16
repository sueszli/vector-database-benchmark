import itertools

def gen(mode, simd, fsize):
    if False:
        print('Hello World!')
    funcname = 'convolution_{mode}_fh{fsize}_{simd}'.format(**vars())
    filename = funcname + '.cpp'
    if simd == 'fma':
        MAX_H = 15 - fsize
    elif simd == 'avx' or simd == 'sse':
        MAX_H = 14 - fsize
    else:
        assert False
    if simd == 'sse':
        width = 4
        mm_type = '__m128'
        mm_load = '_mm_loadu_ps'
        mm_store = '_mm_storeu_ps'
        mm_mul = '_mm_mul_ps'
        mm_add = '_mm_add_ps'
        mm_set1 = '_mm_set1_ps'
        mm_set0 = '_mm_setzero_ps'
        mm_max = '_mm_max_ps'
        mm_set1_sign = ''
        header = ['xmmintrin.h']
    elif simd == 'avx':
        width = 8
        mm_type = '__m256'
        mm_load = '_mm256_loadu_ps'
        mm_store = '_mm256_storeu_ps'
        mm_mul = '_mm256_mul_ps'
        mm_add = '_mm256_add_ps'
        mm_set1 = '_mm256_broadcast_ss'
        mm_set0 = '_mm256_setzero_ps'
        mm_max = '_mm256_max_ps'
        mm_set1_sign = '&'
        header = ['immintrin.h', 'avxintrin.h']
    elif simd == 'fma':
        width = 8
        mm_type = '__m256'
        mm_load = '_mm256_loadu_ps'
        mm_store = '_mm256_storeu_ps'
        mm_set1 = '_mm256_broadcast_ss'
        mm_set0 = '_mm256_setzero_ps'
        mm_max = '_mm256_max_ps'
        mm_set1_sign = '&'
        header = ['immintrin.h', 'avxintrin.h', 'fmaintrin.h']
    with open(filename, 'w') as f:
        for H in range(1, MAX_H + 1):
            f.write('#define SIMD_H{H} do {{ \\\nconst size_t sh = dh; \\\nconst float *src_d = src + sh*src_w; \\\nfloat *dst_d = dst + dh*dst_w; \\\nsize_t dw = dst_w_beg; \\\nfor (; dw < dst_w_end; dw += {width}) {{ \\\n    const size_t sw = dw; \\\n    float *dst_dd = dst_d + dw; \\\n    {mm_type} tmp0; \\\n'.format(**vars()))
            if simd != 'fma':
                f.write('    {mm_type} tmp1; \\\n'.format(**vars()))
            for h in range(H):
                f.write('    {mm_type} res{h}; \\\n    res{h} = {mm_load}(dst_dd + {h}*dst_w); \\\n'.format(**vars()))
            f.write('    for (size_t fw = 0; fw < flt_w; ++fw) {{ \\\n        const float *src_dd = src_d + sw + fw; \\\n'.format(**vars()))
            for fh in range(fsize):
                if mode == 'xcorr':
                    f.write('        {mm_type} vf{fh} = {mm_set1}({mm_set1_sign}filter[{fh}*flt_w+fw]); \\\n'.format(**vars()))
                elif mode == 'conv':
                    f.write('        {mm_type} vf{fh} = {mm_set1}({mm_set1_sign}filter[{fh}*flt_w+flt_w-fw-1]); \\\n'.format(**vars()))
                else:
                    assert False
            for ih in range(H + fsize - 1):
                f.write('        tmp0 = {mm_load}(src_dd + {ih}*src_w); \\\n'.format(**vars()))
                for fh in range(fsize):
                    if mode == 'xcorr':
                        oh = ih - fh
                    elif mode == 'conv':
                        oh = ih - (fsize - fh - 1)
                    else:
                        assert False
                    if oh >= 0 and oh < H:
                        if simd == 'fma':
                            f.write('        res{oh} = _mm256_fmadd_ps(tmp0, vf{fh}, res{oh}); \\\n'.format(**vars()))
                        else:
                            f.write('        tmp1 = {mm_mul}(tmp0, vf{fh}); \\\n'.format(**vars()))
                            f.write('        res{oh} = {mm_add}(res{oh}, tmp1); \\\n'.format(**vars()))
            f.write('    }} \\\n'.format(**vars()))
            for h in range(H):
                f.write('    {mm_store}(dst_dd + {h}*dst_w, res{h}); \\\n'.format(**vars()))
            f.write('}} \\\n}} while (0)\n'.format(**vars()))
            f.write('\n')
        for i in header:
            f.write('#include <{}>\n'.format(i))
        f.write('#include <algorithm>\n\n#include "../convolution_direct_special_cases.h"\n\nnamespace megdnn {{\nnamespace x86 {{\nnamespace detail {{\n\nvoid {funcname}(const float *src, const float *filter, float *dst,\n        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,\n        const size_t flt_w)\n{{\n    (void)src_h;\n    const size_t dst_h_beg = 0;\n    const size_t dst_h_end = dst_h;\n    const size_t dst_w_beg = 0;\n    const size_t dst_w_end = dst_w;\n'.format(**vars()))
        f.write('\n    size_t dh = dst_h_beg;\n    for (; dh + {MAX_H} <= dst_h_end; dh += {MAX_H}) {{\n        SIMD_H{MAX_H};\n    }}\n    switch (dst_h_end - dh) {{\n'.format(**vars()))
        for H in range(1, MAX_H):
            f.write('        case {H}:\n            SIMD_H{H};\n            break;\n'.format(**vars()))
        f.write('    }}\n}}\n\n}} // namespace detail\n}} // namespace x86\n}} // namespace megdnn\n'.format(**vars()))
        for H in range(1, MAX_H + 1):
            f.write('#undef SIMD_H{H}\n'.format(**vars()))

def gen_header(modes, simds, fsizes):
    if False:
        i = 10
        return i + 15
    with open('convolution_direct_special_cases.h', 'w') as f:
        f.write('#pragma once\n\n#include <cstddef>\n#include "megdnn/arch.h"\n\nnamespace megdnn {\nnamespace x86 {\nnamespace detail {\n')
        for (mode, simd, fsize) in itertools.product(modes, simds, fsizes):
            funcname = 'convolution_{mode}_fh{fsize}_{simd}'.format(**vars())
            f.write('\nvoid {funcname}(const float *src, const float *filter, float *dst,\n        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,\n        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("{simd}");\n'.format(**vars()))
        f.write('} // namespace detail\n} // namespace x86\n} // namespace megdnn\n')
if __name__ == '__main__':
    for mode in ['xcorr', 'conv']:
        for fsize in range(1, 8):
            for simd in ['sse', 'avx', 'fma']:
                gen(mode, simd, fsize)
    gen_header(['xcorr', 'conv'], ['sse', 'avx', 'fma'], range(1, 8))