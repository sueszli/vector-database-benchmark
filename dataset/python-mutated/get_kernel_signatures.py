import os
import os.path as osp
import re
import subprocess
import warnings
import pandas as pd

def search_pattern(pattern, file_content):
    if False:
        while True:
            i = 10
    if file_content is not None:
        match_result = re.search(pattern, file_content)
        if match_result is not None:
            return match_result.group(1)
    return None

class KernelSignatureSearcher:
    kernel_sig_pattern = '(template <typename [\\w\\s,]+>[\\s\\n]*void (\\w+Kernel)\\([^\\)]+\\))'
    kernel_reg_pattern = 'PD_REGISTER_KERNEL(_FOR_ALL_DTYPE)?\\([\\s\\n]*(\\w+),[\\s\\n]*(\\w+),[\\s\\n]*(\\w+),[\\s\\n]*([\\w:<>]+)[^\\)]*\\)'
    macro_kernel_reg_pattern = '#define \\w+\\([^\\)]*\\)[\\s\\n\\\\]*PD_REGISTER_KERNEL(_FOR_ALL_DTYPE)?'
    srcs_dir = ['cpu', 'gpu', 'xpu', 'onednn', 'gpudnn', 'kps']
    build_path = None
    filter = {'kernel_name': []}

    def __init__(self, search_path):
        if False:
            print('Hello World!')
        self.search_path = search_path
        self.kernel_func_map = {}
        self.func_signature_map = {}
        self.search_kernel_signature()
        self.search_kernel_registration(search_path)
        self.filter_result()

    @classmethod
    def search(cls, search_path):
        if False:
            i = 10
            return i + 15
        if cls.build_path is None:
            raise ValueError('Please set build_path first.')
        searcher = cls(search_path)
        kernel_func_df = pd.DataFrame(list(searcher.kernel_func_map.items()), columns=['kernel_name', 'kernel_func'])
        func_signature_df = pd.DataFrame(list(searcher.func_signature_map.items()), columns=['kernel_func', 'kernel_signature'])
        return pd.merge(kernel_func_df, func_signature_df, on='kernel_func', how='left')[['kernel_name', 'kernel_signature']].sort_values(by='kernel_name')

    def filter_result(self):
        if False:
            while True:
                i = 10
        for kernel_name in self.filter['kernel_name']:
            if kernel_name in self.kernel_func_map:
                del self.kernel_func_map[kernel_name]

    def search_kernel_signature(self):
        if False:
            print('Hello World!')
        for file in os.listdir(self.search_path):
            if file.endswith('_kernel.h'):
                f = open(osp.join(self.search_path, file), 'r')
                file_content = f.read()
                results = re.findall(self.kernel_sig_pattern, file_content)
                for match_result in results:
                    self.func_signature_map[match_result[1]] = match_result[0]

    def search_kernel_registration(self, path):
        if False:
            return 10
        self.tmp_file_path = osp.join(self.build_path, '.tmp_file.cc')
        self.processed_file_path = self.tmp_file_path.replace('.tmp_file.cc', '.processed_file.cc')
        for file in os.listdir(path):
            file_path = osp.join(path, file)
            if file in self.srcs_dir:
                self.search_kernel_registration(file_path)
            if osp.isdir(file_path):
                continue
            if re.match('\\w+_kernel\\.(cc|cu)', file):
                self._search_kernel_registration(file_path, file)
        if osp.exists(self.processed_file_path):
            os.remove(self.tmp_file_path)
            os.remove(self.processed_file_path)

    def _search_kernel_registration(self, file_path, file):
        if False:
            return 10
        file_content = open(file_path, 'r').read()
        self.header_content = None
        self.file_preprocessed = False
        if re.search(self.macro_kernel_reg_pattern, file_content):
            file_content = self.preprocess_macro(file_content)
            self.file_preprocessed = True
        match_results = re.findall(self.kernel_reg_pattern, file_content)
        for match_result in match_results:
            kernel_name = match_result[1]
            if kernel_name in self.kernel_func_map:
                continue
            kernel_func = match_result[-1].split('<')[0].split('::')[-1]
            self.kernel_func_map[kernel_name] = kernel_func
            if kernel_func in self.func_signature_map:
                continue
            kernel_signature = self.search_target_kernel_signature(kernel_func, file, file_content)
            self.func_signature_map[kernel_func] = kernel_signature
            if kernel_signature is None:
                warnings.warn("Can't find kernel signature for kernel: " + kernel_func + ', which is registered in file: ' + file_path)

    def search_target_kernel_signature(self, kernel_func, file, file_content):
        if False:
            while True:
                i = 10
        target_kernel_signature_pattern = self.kernel_sig_pattern.replace('(\\w+Kernel)', kernel_func)
        kernel_signature = search_pattern(target_kernel_signature_pattern, file_content)
        if kernel_signature is not None:
            return kernel_signature
        if not self.file_preprocessed:
            file_content = self.preprocess_macro(file_content)
            kernel_signature = search_pattern(target_kernel_signature_pattern, file_content)
            if kernel_signature is not None:
                return kernel_signature
        if self.header_content is None:
            header_path = osp.join(self.search_path, file.split('.')[0] + '.h')
            if osp.exists(header_path):
                self.header_content = open(header_path, 'r').read()
        if self.header_content is not None:
            self.header_content = self.preprocess_macro(self.header_content)
            kernel_signature = search_pattern(target_kernel_signature_pattern, self.header_content)
            if kernel_signature is not None:
                return kernel_signature
        return None

    def preprocess_macro(self, file_content):
        if False:
            for i in range(10):
                print('nop')
        if file_content is None:
            return file_content
        file_content = re.sub('#(include|pragma)', '// \\g<0>', file_content)
        with open(self.tmp_file_path, 'w') as f:
            f.write(file_content)
        subprocess.run(['g++', '-E', self.tmp_file_path, '-o', self.processed_file_path])
        subprocess.run(['clang-format', '-i', self.processed_file_path])
        file_content = open(self.processed_file_path, 'r').read()
        return file_content

def get_kernel_signatures():
    if False:
        for i in range(10):
            print('nop')
    '\n    Get kernel signatures of all kernels registered in phi/kernels, and\n    generate a csv file named \'kernel_signatures.csv\' in Paddle/build.\n\n    If you want to filter some kernels in result, you can add them to\n    KernelSignatureSearcher.filter["kernel_name"].\n    '
    Paddle_path = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
    build_path = osp.join(Paddle_path, 'build')
    os.makedirs(build_path, exist_ok=True)
    KernelSignatureSearcher.build_path = build_path
    base_path = osp.join(Paddle_path, 'paddle/phi/kernels')
    kernel_signature_df = KernelSignatureSearcher.search(base_path)
    independent_subdir = ['fusion', 'sparse', 'strings']
    for subdir in independent_subdir:
        sub_path = osp.join(base_path, subdir)
        sub_df = KernelSignatureSearcher.search(sub_path)
        kernel_signature_df = pd.concat([kernel_signature_df, sub_df], ignore_index=True)
    output_csv_path = osp.join(build_path, 'kernel_signatures.csv')
    kernel_signature_df.to_csv(output_csv_path, index=False)
    print(kernel_signature_df)
if __name__ == '__main__':
    get_kernel_signatures()