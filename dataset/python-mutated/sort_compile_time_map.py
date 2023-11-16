from click import command
import subprocess
import re
import tempfile

def topN_file(src_file_path, des_file_path, N):
    if False:
        while True:
            i = 10
    src = open(src_file_path)
    des = open(des_file_path, 'w')
    lines = src.readlines()
    file_map = {}
    for (index, line) in enumerate(lines):
        if 'Building C' in line and 'megdnn.dir' in line:
            if 'Elapsed time: ' in lines[index + 1]:
                key = line[line.find('megdnn.dir'):line.find('.o')]
                value = lines[index + 1][lines[index + 1].find('Elapsed time: ') + 14:lines[index + 1].find(' s. ')]
                file_map[key] = value
    a = sorted(file_map.items(), key=lambda x: int(x[1]), reverse=True)
    result_file = a[:N]
    result_opr = []
    for i in result_file:
        key = '/'.join(list(re.findall('megdnn.dir\\/(.*?)\\/(.*?)[\\.\\/]', i[0])[0]))
        if key not in result_opr:
            result_opr.append(key)
            des.write(key + '\n')
    src.close()
    des.close()
    return result_opr

def compile(cmd: str, dir: str, failed_name=3):
    if False:
        return 10
    for i in range(failed_name):
        subprocess.run(cmd, shell=True, cwd=t)
if __name__ == '__main__':
    cmd = f'\ncmake .. -DMGE_PROFILE_COMPILE_TIME=ON\ntime make -j1 megdnn | tee megdnn_map_compile_time.txt\n'
    with tempfile.TemporaryDirectory(dir='../../../../') as t:
        compile(cmd, t)
        topN_file(t + '/megdnn_map_compile_time.txt', './priority_compile_opr.txt', 500)