"""upload_gtest.py v0.1.0 -- uploads a Google Test patch for review.

This simple wrapper passes all command line flags and
--cc=googletestframework@googlegroups.com to upload.py.

USAGE: upload_gtest.py [options for upload.py]
"""
__author__ = 'wan@google.com (Zhanyong Wan)'
import os
import sys
CC_FLAG = '--cc='
GTEST_GROUP = 'googletestframework@googlegroups.com'

def main():
    if False:
        for i in range(10):
            print('nop')
    my_dir = os.path.dirname(os.path.abspath(__file__))
    upload_py_path = os.path.join(my_dir, 'upload.py')
    upload_py_argv = [upload_py_path]
    found_cc_flag = False
    for arg in sys.argv[1:]:
        if arg.startswith(CC_FLAG):
            found_cc_flag = True
            cc_line = arg[len(CC_FLAG):]
            cc_list = [addr for addr in cc_line.split(',') if addr]
            if GTEST_GROUP not in cc_list:
                cc_list.append(GTEST_GROUP)
            upload_py_argv.append(CC_FLAG + ','.join(cc_list))
        else:
            upload_py_argv.append(arg)
    if not found_cc_flag:
        upload_py_argv.append(CC_FLAG + GTEST_GROUP)
    os.execv(upload_py_path, upload_py_argv)
if __name__ == '__main__':
    main()