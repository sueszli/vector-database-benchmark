import subprocess
import sys

def main():
    if False:
        for i in range(10):
            print('nop')
    path = sys.argv[1]
    out_path = sys.argv[2]
    llvm_config = sys.argv[3]
    srcs = []
    srcs.append('#include <absl/strings/string_view.h>')
    srcs.append('namespace cinn::backends {')
    srcs.append('static const absl::string_view kRuntimeLlvmIr(')
    srcs.append('R"ROC(')
    with open(path, 'r') as fr:
        srcs.append(fr.read())
    srcs.append(')ROC"')
    srcs.append(');\n')
    cmd = f'{llvm_config} --version'
    version = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split('.')
    srcs.append('struct llvm_version {')
    for (v, n) in zip(['major', 'minor', 'micro'], version):
        srcs.append('  static constexpr int k{} = {};'.format(v.title(), ''.join(filter(str.isdigit, n))))
    srcs.append('};')
    srcs.append('}  // namespace cinn::backends')
    with open(out_path, 'w') as fw:
        fw.write('\n'.join(srcs))

def get_clang_version():
    if False:
        for i in range(10):
            print('nop')
    pass
if __name__ == '__main__':
    main()