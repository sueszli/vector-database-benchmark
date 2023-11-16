from executor import run_fuzzer_on_test_file
TEST_FILE_LIST = ['Lib/ctypes/test/test_anon.py', 'Lib/ctypes/test/test_array_in_pointer.py', 'Lib/ctypes/test/test_bitfields.py', 'Lib/ctypes/test/test_callbacks.py', 'Lib/test/test_fileio.py', 'Lib/test/test_memoryview.py', 'Tools/fuzzer/tests/verifier_test.py']
SUBPROCESSES = 4

def run_fuzzer_on_test_files():
    if False:
        i = 10
        return i + 15
    with open('fuzzer_output.txt', 'w+') as outfile:
        for i in TEST_FILE_LIST:
            run_fuzzer_on_test_file(i, SUBPROCESSES, outfile)
    with open('fuzzer_output.txt', 'r') as outfile:
        print(outfile.read())
if __name__ == '__main__':
    run_fuzzer_on_test_files()