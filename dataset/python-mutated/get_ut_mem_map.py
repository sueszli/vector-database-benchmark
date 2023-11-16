import json
import os
import sys

def get_ut_mem(rootPath):
    if False:
        for i in range(10):
            print('nop')
    case_dic = {}
    for (parent, dirs, files) in os.walk(rootPath):
        for f in files:
            if f.endswith('$-gpu.log'):
                continue
            ut = f.replace('^', '').replace('$.log', '')
            case_dic[ut] = {}
            filename = f'{parent}/{f}'
            fi = open(filename, mode='rb')
            lines = fi.readlines()
            mem_reserved1 = -1
            mem_nvidia1 = -1
            caseTime = -1
            for line in lines:
                line = line.decode('utf-8', errors='ignore')
                if '[Memory Usage (Byte)] gpu' in line:
                    mem_reserved = round(float(line.split(' : Reserved = ')[1].split(', Allocated = ')[0]), 2)
                    if mem_reserved > mem_reserved1:
                        mem_reserved1 = mem_reserved
                if 'MAX_GPU_MEMORY_USE=' in line:
                    mem_nvidia = round(float(line.split('MAX_GPU_MEMORY_USE=')[1].split('\\n')[0].strip()), 2)
                    if mem_nvidia > mem_nvidia1:
                        mem_nvidia1 = mem_nvidia
                if 'Total Test time (real)' in line:
                    caseTime = float(line.split('Total Test time (real) =')[1].split('sec')[0].strip())
            if mem_reserved1 != -1:
                case_dic[ut]['mem_reserved'] = mem_reserved1
            if mem_nvidia1 != -1:
                case_dic[ut]['mem_nvidia'] = mem_nvidia1
            if caseTime != -1:
                case_dic[ut]['time'] = caseTime
            fi.close()
    if not os.path.exists('/pre_test'):
        os.mkdir('/pre_test')
    ut_mem_map_file = '/pre_test/ut_mem_map.json'
    with open(ut_mem_map_file, 'w') as f:
        json.dump(case_dic, f)
if __name__ == '__main__':
    rootPath = sys.argv[1]
    get_ut_mem(rootPath)