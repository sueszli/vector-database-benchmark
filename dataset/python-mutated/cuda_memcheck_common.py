class ParseError(Exception):
    """Whenever the simple parser is unable to parse the report, this exception will be raised"""
    pass

class Report:
    """A report is a container of errors, and a summary on how many errors are found"""

    def __init__(self, text, errors):
        if False:
            return 10
        self.text = text
        self.num_errors = int(text.strip().split()[2])
        self.errors = errors
        if len(errors) != self.num_errors:
            if len(errors) == 10000 and self.num_errors > 10000:
                self.num_errors = 10000
            else:
                raise ParseError('Number of errors does not match')

class Error:
    """Each error is a section in the output of cuda-memcheck.
    Each error in the report has an error message and a backtrace. It looks like:

    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaGetLastError + 0x163) [0x4c493]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x5b77a05]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x39d6d1d]
    =========     .....
    """

    def __init__(self, lines):
        if False:
            for i in range(10):
                print('nop')
        self.message = lines[0]
        lines = lines[2:]
        self.stack = [l.strip() for l in lines]

def parse(message):
    if False:
        i = 10
        return i + 15
    'A simple parser that parses the report of cuda-memcheck. This parser is meant to be simple\n    and it only split the report into separate errors and a summary. Where each error is further\n    splitted into error message and backtrace. No further details are parsed.\n\n    A report contains multiple errors and a summary on how many errors are detected. It looks like:\n\n    ========= CUDA-MEMCHECK\n    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaPointerGetAttributes.\n    =========     Saved host backtrace up to driver entry point at error\n    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]\n    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaPointerGetAttributes + 0x1a9) [0x428b9]\n    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x5b778a9]\n    =========     .....\n    =========\n    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.\n    =========     Saved host backtrace up to driver entry point at error\n    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]\n    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaGetLastError + 0x163) [0x4c493]\n    =========     .....\n    =========\n    ========= .....\n    =========\n    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.\n    =========     Saved host backtrace up to driver entry point at error\n    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]\n    =========     .....\n    =========     Host Frame:python (_PyEval_EvalFrameDefault + 0x6a0) [0x1d0ad0]\n    =========     Host Frame:python (_PyEval_EvalCodeWithName + 0xbb9) [0x116db9]\n    =========\n    ========= ERROR SUMMARY: 4 errors\n    '
    errors = []
    HEAD = '========='
    headlen = len(HEAD)
    started = False
    in_message = False
    message_lines = []
    lines = message.splitlines()
    for l in lines:
        if l == HEAD + ' CUDA-MEMCHECK':
            started = True
            continue
        if not started or not l.startswith(HEAD):
            continue
        l = l[headlen + 1:]
        if l.startswith('ERROR SUMMARY:'):
            return Report(l, errors)
        if not in_message:
            in_message = True
            message_lines = [l]
        elif l == '':
            errors.append(Error(message_lines))
            in_message = False
        else:
            message_lines.append(l)
    raise ParseError('No error summary found')