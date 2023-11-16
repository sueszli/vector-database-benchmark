def gen_forward():
    if False:
        for i in range(10):
            print('nop')
    kernels = [3, 5, 7, 15, 31, 63, 127, 255]
    seqs = [32 * x for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
    head = '\n/**\n * Copyright (c) Facebook, Inc. and its affiliates.\n *\n * This source code is licensed under the MIT license found in the\n * LICENSE file in the root directory of this source tree.\n */\n\n#include "lightconv_cuda.cuh"\n\nstd::vector<at::Tensor> lightconv_cuda_forward(at::Tensor input, at::Tensor filters, int padding_l) {\n\n    at::DeviceGuard g(input.device());\n    const auto minibatch = input.size(0);\n    const auto numFeatures = input.size(1);\n    const auto sequenceLength = input.size(2);\n\n    const auto numHeads = filters.size(0);\n    const auto filterSize = filters.size(1);\n\n    const auto numFiltersInBlock = numFeatures / numHeads;\n\n    const dim3 blocks(minibatch, numFeatures);\n\n    auto output = at::zeros_like(input);\n    auto stream = at::cuda::getCurrentCUDAStream();\n'
    sequence_if = '\n    if (sequenceLength <= {seq}) {{\n        switch(filterSize) {{\n'
    case_k = '\n            case {k}:\n'
    main_block = '\n                if (padding_l == {pad}) {{\n                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {{\n                        lightconv_forward_kernel<{k}, {b_size}, {pad}, scalar_t>\n                        <<<blocks, {b_size}, 0, stream>>>(\n                                input.data<scalar_t>(),\n                                filters.data<scalar_t>(),\n                                minibatch,\n                                sequenceLength,\n                                numFeatures,\n                                numFiltersInBlock,\n                                output.data<scalar_t>());\n                    }}));\n                }} else\n'
    bad_padding = '\n                {\n                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;\n                }\n                break;\n'
    bad_filter = '\n            default:\n                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;\n        }\n'
    con_else = '\n    } else\n'
    final_else = '\n    {\n        switch(filterSize) {\n'
    final_return = '\n    }\n\n    return {output};\n}\n'
    with open('lightconv_cuda_forward.cu', 'w') as forward:
        forward.write(head)
        for seq in seqs:
            forward.write(sequence_if.format(seq=seq))
            for k in kernels:
                forward.write(case_k.format(k=k))
                for pad in [k // 2, k - 1]:
                    forward.write(main_block.format(k=k, b_size=seq, pad=pad))
                forward.write(bad_padding)
            forward.write(bad_filter)
            forward.write(con_else)
        forward.write(final_else)
        for k in kernels:
            forward.write(case_k.format(k=k))
            for pad in [k // 2, k - 1]:
                forward.write(main_block.format(k=k, b_size=seq, pad=pad))
            forward.write(bad_padding)
        forward.write(bad_filter)
        forward.write(final_return)

def gen_backward():
    if False:
        while True:
            i = 10
    head = '\n/**\n * Copyright (c) Facebook, Inc. and its affiliates.\n *\n * This source code is licensed under the MIT license found in the\n * LICENSE file in the root directory of this source tree.\n */\n\n#include "lightconv_cuda.cuh"\n\nstd::vector<at::Tensor> lightconv_cuda_backward(\n        at::Tensor gradOutput,\n        int padding_l,\n        at::Tensor input,\n        at::Tensor filters) {\n\n    // gradWrtInput\n    const int minibatch = input.size(0);\n    const int numFeatures = input.size(1);\n    const int sequenceLength = input.size(2);\n\n    const int numHeads = filters.size(0);\n    const int filterSize = filters.size(1);\n\n    const dim3 gradBlocks(minibatch, numFeatures);\n    const dim3 weightGradFirstpassShortBlocks(minibatch, numHeads);\n    const dim3 weightGradSecondpassBlocks(numHeads, filterSize);\n\n    const int numFiltersInBlock = numFeatures / numHeads;\n\n    auto gradInput = at::zeros_like(input);\n    auto gradFilters = at::zeros_like(filters);\n\n    at::DeviceGuard g(input.device());\n    auto stream = at::cuda::getCurrentCUDAStream();\n\n    switch(filterSize) {\n'
    sequence_if = '\n            if (sequenceLength <= {seq}) {{\n'
    case_k = '\n        case {k}:\n'
    main_block = '\n                if (padding_l == {p}) {{\n                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {{\n                        lightconv_grad_wrt_input_kernel<{k}, {b_size}, {p}, scalar_t>\n                        <<<gradBlocks, {b_size}, 0, stream>>>(\n                                gradOutput.data<scalar_t>(),\n                                filters.data<scalar_t>(),\n                                minibatch,\n                                sequenceLength,\n                                numFeatures,\n                                numFiltersInBlock,\n                                gradInput.data<scalar_t>());\n\n'
    weight_grad_short = '\n                        at::Tensor tempSumGradFilters = at::zeros({{minibatch, numHeads, filterSize}}, input.options().dtype(at::kFloat));\n                        lightconv_grad_wrt_weights_firstpass_short_kernel<{k}, {b_size}, {p}, scalar_t>\n                        <<<weightGradFirstpassShortBlocks, {b_size}, 0, stream>>>(\n                                input.data<scalar_t>(),\n                                gradOutput.data<scalar_t>(),\n                                minibatch,\n                                sequenceLength,\n                                numFeatures,\n                                numFiltersInBlock,\n                                numHeads,\n                                tempSumGradFilters.data<float>()\n                        );\n\n                        lightconv_grad_wrt_weights_secondpass_short_kernel<{k}, {b_size}, scalar_t>\n                        <<<weightGradSecondpassBlocks, {b_size}, 0, stream>>>(\n                                tempSumGradFilters.data<float>(),\n                                minibatch,\n                                numFiltersInBlock,\n                                gradFilters.data<scalar_t>()\n                        );\n                    }}));\n                }} else\n'
    weight_grad = '\n                        at::Tensor tempSumGradFilters = at::zeros({{minibatch, numFeatures, filterSize}}, input.options().dtype(at::kFloat));\n                        lightconv_grad_wrt_weights_firstpass_kernel<{k}, {b_size}, {p}, scalar_t>\n                        <<<gradBlocks, {b_size}, 0, stream>>>(\n                                input.data<scalar_t>(),\n                                gradOutput.data<scalar_t>(),\n                                minibatch,\n                                sequenceLength,\n                                numFeatures,\n                                numFiltersInBlock,\n                                tempSumGradFilters.data<float>()\n                        );\n\n                        lightconv_grad_wrt_weights_secondpass_kernel<{k}, {b_size}, scalar_t>\n                        <<<weightGradSecondpassBlocks, {b_size}, 0, stream>>>(\n                                tempSumGradFilters.data<float>(),\n                                minibatch,\n                                numFiltersInBlock,\n                                gradFilters.data<scalar_t>()\n                        );\n                    }}));\n                }} else\n'
    bad_padding = '\n                {\n                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;\n                }\n'
    breakout = '\n                break;\n'
    bad_filter = '\n        default:\n            std::cout << "WARNING: Unsupported filter length passed - skipping backward pass" << std::endl;\n'
    con_else = '\n            } else\n'
    final_else = '\n    {\n        switch(filterSize) {\n'
    last_return = '\n    }\n    return {gradInput, gradFilters};\n}\n'
    kernels = [3, 5, 7, 15, 31, 63, 127, 255]
    seqs = [32 * x for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
    thresh = [32, 32, 64, 128, 256, -1, -1, -1]
    max_mem = [-1, -1, -1, -1, -1, 192, 96, 64]
    with open('lightconv_cuda_backward.cu', 'w') as backward:
        backward.write(head)
        for (k, t, mem) in zip(kernels, thresh, max_mem):
            backward.write(case_k.format(k=k))
            for seq in seqs:
                if (t == -1 or seq <= t) and (mem == -1 or seq < mem):
                    backward.write(sequence_if.format(seq=seq))
                    for p in [k // 2, k - 1]:
                        backward.write(main_block.format(k=k, b_size=seq, p=p))
                        backward.write(weight_grad_short.format(k=k, b_size=seq, p=p))
                    backward.write(bad_padding)
                else:
                    for p in [k // 2, k - 1]:
                        backward.write(main_block.format(k=k, b_size=32, p=p))
                        backward.write(weight_grad.format(k=k, b_size=32, p=p))
                    backward.write(bad_padding)
                    backward.write(breakout)
                    break
                backward.write(con_else)
        backward.write(bad_filter)
        backward.write(last_return)
if __name__ == '__main__':
    gen_forward()
    gen_backward()