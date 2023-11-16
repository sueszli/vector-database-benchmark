from ..tool import clang_coverage
from ..util.setting import CompilerType, Option, TestList, TestPlatform
from ..util.utils import check_compiler_type
from .init import detect_compiler_type
from .run import clang_run, gcc_run

def get_json_report(test_list: TestList, options: Option) -> None:
    if False:
        for i in range(10):
            print('nop')
    cov_type = detect_compiler_type()
    check_compiler_type(cov_type)
    if cov_type == CompilerType.CLANG:
        if options.need_run:
            clang_run(test_list)
        if options.need_merge:
            clang_coverage.merge(test_list, TestPlatform.OSS)
        if options.need_export:
            clang_coverage.export(test_list, TestPlatform.OSS)
    elif cov_type == CompilerType.GCC:
        if options.need_run:
            gcc_run(test_list)