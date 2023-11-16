from __future__ import print_function
import os, sys, importlib
import shutil, time, datetime
import subprocess
from cntk_helpers import getFilesInDirectory
import PARAMETERS

def run_fastrcnn_with_config_file(cntkBuildPath='cntk'):
    if False:
        while True:
            i = 10
    p = PARAMETERS.get_parameters_for_dataset()
    cntkCmdStrPattern = cntkBuildPath + ' configFile={0}/fastrcnn.cntk currentDirectory={0} {1}'
    NumLabels = p.nrClasses
    NumTrainROIs = p.cntk_nrRois
    TrainROIDim = p.cntk_nrRois * 4
    TrainROILabelDim = p.cntk_nrRois * p.nrClasses
    NumTestROIs = p.cntk_nrRois
    TestROIDim = p.cntk_nrRois * 4
    TestROILabelDim = p.cntk_nrRois * p.nrClasses
    cntk_args = 'NumLabels={} NumTrainROIs={}'.format(NumLabels, NumTrainROIs)
    cntk_args += ' TrainROIDim={} TrainROILabelDim={}'.format(TrainROIDim, TrainROILabelDim)
    cntk_args += ' NumTestROIs={}'.format(NumTestROIs)
    cntk_args += ' TestROIDim={} TestROILabelDim={}'.format(TestROIDim, TestROILabelDim)
    shutil.copy(os.path.join(p.cntkTemplateDir, 'fastrcnn.cntk'), p.cntkFilesDir)
    tstart = datetime.datetime.now()
    cmdStr = cntkCmdStrPattern.format(p.cntkFilesDir, cntk_args)
    print(cmdStr)
    pid = subprocess.Popen(cmdStr.split(' '), cwd=p.cntkFilesDir)
    pid.wait()
    print('Time running cntk [s]: ' + str((datetime.datetime.now() - tstart).total_seconds()))
    modelDir = os.path.join(p.cntkFilesDir, 'Output')
    filenames = getFilesInDirectory(modelDir, postfix=None)
    for filename in filenames:
        if 'Fast-RCNN.model.' in filename:
            os.remove(os.path.join(modelDir, filename))
    assert pid.returncode == 0, 'ERROR: cntk ended with exit code {}'.format(pid.returncode)
    print('DONE.')
    return True
if __name__ == '__main__':
    run_fastrcnn_with_config_file()