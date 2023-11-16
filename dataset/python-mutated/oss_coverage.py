import time
from package.oss.cov_json import get_json_report
from package.oss.init import initialization
from package.tool.summarize_jsons import summarize_jsons
from package.util.setting import TestPlatform
from package.util.utils import print_time

def report_coverage() -> None:
    if False:
        for i in range(10):
            print('nop')
    start_time = time.time()
    (options, test_list, interested_folders) = initialization()
    get_json_report(test_list, options)
    if options.need_summary:
        summarize_jsons(test_list, interested_folders, [''], TestPlatform.OSS)
    print_time('Program Total Time: ', start_time)
if __name__ == '__main__':
    report_coverage()