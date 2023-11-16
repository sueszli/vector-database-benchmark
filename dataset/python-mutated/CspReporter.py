import argparse
import glob
import os
from multiprocessing import Process
from CspFileReader import DCGM_PATH, FILEORGANIZEFORM_BYRANK, FILEORGANIZEFORM_BYTRAINER, NET_PATH, PROFILE_PATH, TIME_PATH, getLogger
from DCGMFileReader import dcgmFileReader
from ProfileFileReader import profileFileReader

def get_argparse():
    if False:
        return 10
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile_path', type=str, default='.', help='Working path that store the monitor data.')
    parser.add_argument('--timeline_path', type=str, default='.', help='Output timeline file name.')
    parser.add_argument('--gpuPerTrainer', type=int, default=8, help='Gpus per trainer.')
    parser.add_argument('--trainerNum', type=int, default=4, help='Num of trainer.')
    parser.add_argument('--groupSize', type=int, default=8, help='Num of trainer in a group.')
    parser.add_argument('--displaySize', type=int, default=2, help='Num of line need to display in a group.')
    return parser.parse_args()

class CspReporter:

    def __init__(self, args):
        if False:
            return 10
        self._args = args
        print(self._args)
        self._workPath = self._args.profile_path
        self._saveFilePath = self._args.timeline_path
        self._gpuPerTrainer = self._args.gpuPerTrainer
        self._groupSize = self._args.groupSize
        self._displaySize = self._args.displaySize
        self._trainerNum = self._args.trainerNum
        self._checkArgs()
        self._init_logger()
        self._init_timeInfo()
        self._init_reader()

    def _checkArgs(self):
        if False:
            for i in range(10):
                print('nop')
        if self._trainerNum % self._groupSize != 0:
            raise Exception('Input args error: trainerNum[%d] %% groupSize[%d] != 0' % (self._trainerNum, self._groupSize))

    def _init_logger(self):
        if False:
            return 10
        self._logger = getLogger()

    def _init_reader(self):
        if False:
            i = 10
            return i + 15
        self._dcgmPath = os.path.join(self._workPath, DCGM_PATH)
        self._netPath = os.path.join(self._workPath, NET_PATH)
        self._profilePath = os.path.join(self._workPath, PROFILE_PATH)
        self._netFileReaderArgs = {'dataPath': self._netPath, 'groupSize': self._groupSize, 'displaySize': self._displaySize, 'gpuPerTrainer': self._gpuPerTrainer, 'minTimeStamp': self._minTimeStamp, 'organizeForm': FILEORGANIZEFORM_BYTRAINER}
        self._dcgmFileReaderArgs = {'dataPath': self._dcgmPath, 'groupSize': self._groupSize, 'displaySize': self._displaySize, 'gpuPerTrainer': self._gpuPerTrainer, 'minTimeStamp': self._minTimeStamp, 'organizeForm': FILEORGANIZEFORM_BYTRAINER}
        self._profileFileReaderArgs = {'dataPath': self._profilePath, 'groupSize': self._groupSize, 'displaySize': self._displaySize, 'gpuPerTrainer': self._gpuPerTrainer, 'minTimeStamp': self._minTimeStamp, 'organizeForm': FILEORGANIZEFORM_BYRANK}
        self._dcgmFileReader = dcgmFileReader(self._logger, self._dcgmFileReaderArgs)
        self._profileFileReader = profileFileReader(self._logger, self._profileFileReaderArgs)

    def _init_timeInfo(self):
        if False:
            i = 10
            return i + 15
        self._timePath = os.path.join(self._workPath, TIME_PATH)
        self._timeInfo = {}
        self._minTimeStamp = 0
        self._set_timeInfo()

    def _set_timeInfo(self, timeFileNamePrefix='time.txt', sed='.'):
        if False:
            for i in range(10):
                print('nop')
        timeFileNameList = glob.glob(os.path.join(self._timePath, timeFileNamePrefix, sed, '*'))
        for timeFileName in timeFileNameList:
            trainerId = int(timeFileName.split(sed)[-1])
            gpuId = int(timeFileName.split(sed)[-2])
            info = {}
            with open(timeFileName, 'r') as rf:
                for line in rf:
                    if line.startswith('start time:'):
                        info['start_time'] = int(float(line.split(':')[-1]) * 1000000000.0)
                        self._minTimeStamp = min(self._minTimeStamp, info['start_time'])
                    if line.startswith('end time:'):
                        info['end_time'] = int(float(line.split(':')[-1]) * 1000000000.0)
            if not info:
                self._timeInfo[gpuId * trainerId] = info

    def _generateTraceFileByGroupAndGpuId(self, pipileInfo, netInfo, groupId, gpuId):
        if False:
            i = 10
            return i + 15
        dcgmInfoDict = self._dcgmFileReader.getDcgmInfoDict(groupId, gpuId)
        opInfoDict = self._profileFileReader.getOpInfoDict(groupId, gpuId)
        traceObj = {}
        traceObj['traceEvents'] = pipileInfo[str(gpuId)] + opInfoDict['traceEvents'] + dcgmInfoDict['traceEvents'] + netInfo['traceEvents']
        self._profileFileReader.dumpDict(traceObj, 'traceFile', groupId, gpuId, False, self._saveFilePath)

    def _generateTraceFileByGroup(self, groupId, processNum):
        if False:
            for i in range(10):
                print('nop')
        pipileInfo = self._profileFileReader.getPipeLineInfo(groupId, processNum)
        dcgmInfo = self._dcgmFileReader.getDCGMTraceInfo(groupId, processNum)
        netInfo = {}
        netInfo['traceEvents'] = []
        opInfo = self._profileFileReader.getOPTraceInfo(groupId)
        processPool = []
        pidList = []
        for gpuId in range(self._gpuPerTrainer):
            subproc = Process(target=self._generateTraceFileByGroupAndGpuId, args=(pipileInfo, netInfo, groupId, gpuId))
            processPool.append(subproc)
            subproc.start()
            pidList.append(subproc.pid)
            self._logger.info('[traceFile]: process [%d] has been started, total task num is %d ...' % (subproc.pid, 1))
        for t in processPool:
            t.join()
            pidList.remove(t.pid)
            self._logger.info('[traceFile]: process [%d] has exited! remained %d process!' % (t.pid, len(pidList)))

    def generateTraceFile(self, processNum=8):
        if False:
            i = 10
            return i + 15
        processPool = []
        pidList = []
        for groupId in range(self._trainerNum / self._groupSize):
            subproc = Process(target=self._generateTraceFileByGroup, args=(groupId, processNum))
            processPool.append(subproc)
            subproc.start()
            pidList.append(subproc.pid)
            self._logger.info('[GroupTraceFile]: process [%d] has been started, total task num is %d ...' % (subproc.pid, 1))
        for t in processPool:
            t.join()
            pidList.remove(t.pid)
            self._logger.info('[GroupTraceFile]: process [%d] has exited! remained %d process!' % (t.pid, len(pidList)))
if __name__ == '__main__':
    args = get_argparse()
    tl = CspReporter(args)
    tl.generateTraceFile()