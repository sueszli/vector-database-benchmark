import json
import os
import subprocess
import sys
import time
import traceback
from xml.dom import minidom

def main(argv):
    if False:
        i = 10
        return i + 15
    metrics_output_dir = os.environ['METRIC_OUTPUT_DIR']
    cmd = 'nvidia-smi -q -x'.split()
    retry = 0
    while True:
        smi = subprocess.run(cmd, timeout=20, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if smi.returncode != 0:
            retry += 1
            print(f'gpu_metrics_collector error: nvidia-smi return code is {smi.returncode}', file=sys.stderr)
            print('=' * 20 + f'\nCaptured stdout: {smi.stdout}', file=sys.stderr)
            print('=' * 20 + f'\nCaptured stderr: {smi.stderr}', file=sys.stderr)
            gen_empty_gpu_metric(metrics_output_dir)
            if retry >= 5:
                break
        else:
            parse_nvidia_smi_result(smi.stdout, metrics_output_dir)
        time.sleep(5)

def parse_nvidia_smi_result(smi, outputDir):
    if False:
        for i in range(10):
            print('nop')
    old_umask = os.umask(0)
    try:
        xmldoc = minidom.parseString(smi)
        gpuList = xmldoc.getElementsByTagName('gpu')
        with open(os.path.join(outputDir, 'gpu_metrics'), 'a') as outputFile:
            outPut = {}
            outPut['Timestamp'] = time.asctime(time.localtime())
            outPut['gpuCount'] = len(gpuList)
            outPut['gpuInfos'] = []
            for (gpuIndex, gpu) in enumerate(gpuList):
                gpuInfo = {}
                gpuInfo['index'] = gpuIndex
                gpuInfo['gpuUtil'] = gpu.getElementsByTagName('utilization')[0].getElementsByTagName('gpu_util')[0].childNodes[0].data.replace('%', '').strip()
                gpuInfo['gpuMemUtil'] = gpu.getElementsByTagName('utilization')[0].getElementsByTagName('memory_util')[0].childNodes[0].data.replace('%', '').strip()
                processes = gpu.getElementsByTagName('processes')
                runningProNumber = len(processes[0].getElementsByTagName('process_info'))
                gpuInfo['activeProcessNum'] = runningProNumber
                outPut['gpuInfos'].append(gpuInfo)
            print(outPut)
            outputFile.write('{}\n'.format(json.dumps(outPut, sort_keys=True)))
            outputFile.flush()
    except Exception as error:
        print('gpu_metrics_collector error: %s' % error)
    finally:
        os.umask(old_umask)

def gen_empty_gpu_metric(outputDir):
    if False:
        print('Hello World!')
    old_umask = os.umask(0)
    try:
        with open(os.path.join(outputDir, 'gpu_metrics'), 'a') as outputFile:
            outPut = {}
            outPut['Timestamp'] = time.asctime(time.localtime())
            outPut['gpuCount'] = 0
            outPut['gpuInfos'] = []
            print(outPut)
            outputFile.write('{}\n'.format(json.dumps(outPut, sort_keys=True)))
            outputFile.flush()
    except Exception:
        traceback.print_exc()
    finally:
        os.umask(old_umask)
if __name__ == '__main__':
    main(sys.argv[1:])