"""
Used to prepare simulated data.
"""
import math
import os.path
import queue
import threading
import numpy as np
import torch
from modelscope.utils.logger import get_logger
logger = get_logger()
BLOCK_DEC = 2
BLOCK_CAT = 3
FBANK_SIZE = 40
LABEL_SIZE = 1
LABEL_GAIN = 100.0

class KWSDataset:
    """
    dataset for keyword spotting and vad
    conf_basetrain:         basetrain configure file path
    conf_finetune:          finetune configure file path, null allowed
    numworkers:             no. of workers
    basetrainratio:         basetrain workers ratio
    numclasses:             no. of nn output classes, 2 classes to generate vad label
    blockdec:               block decimation
    blockcat:               block concatenation
    """

    def __init__(self, conf_basetrain, conf_finetune, numworkers, basetrainratio, numclasses, blockdec=BLOCK_CAT, blockcat=BLOCK_CAT):
        if False:
            return 10
        super().__init__()
        self.numclasses = numclasses
        self.blockdec = blockdec
        self.blockcat = blockcat
        self.sims_base = []
        self.sims_senior = []
        self.setup_sims(conf_basetrain, conf_finetune, numworkers, basetrainratio)

    def release(self):
        if False:
            print('Hello World!')
        for sim in self.sims_base:
            del sim
        for sim in self.sims_senior:
            del sim
        del self.base_conf
        del self.senior_conf
        logger.info('KWSDataset: Released.')

    def setup_sims(self, conf_basetrain, conf_finetune, numworkers, basetrainratio):
        if False:
            print('Hello World!')
        if not os.path.exists(conf_basetrain):
            raise ValueError(f'{conf_basetrain} does not exist!')
        if not os.path.exists(conf_finetune):
            raise ValueError(f'{conf_finetune} does not exist!')
        import py_sound_connect
        logger.info('KWSDataset init SoundConnect...')
        num_base = math.ceil(numworkers * basetrainratio)
        num_senior = numworkers - num_base
        self.base_conf = py_sound_connect.ConfigFile(conf_basetrain)
        self.senior_conf = py_sound_connect.ConfigFile(conf_finetune)
        for i in range(num_base):
            fs = py_sound_connect.FeatSimuKWS(self.base_conf.params)
            self.sims_base.append(fs)
        for i in range(num_senior):
            self.sims_senior.append(py_sound_connect.FeatSimuKWS(self.senior_conf.params))
        logger.info('KWSDataset init SoundConnect finished.')

    def getBatch(self, id):
        if False:
            while True:
                i = 10
        '\n        Generate a data batch\n\n        Args:\n            id: worker id\n\n        Return: time x channel x feature, label\n        '
        fs = self.get_sim(id)
        fs.processBatch()
        featsize = fs.featSize()
        labelsize = fs.labelSize()
        numchs = featsize // FBANK_SIZE
        fs_feat = fs.feat()
        data = np.frombuffer(fs_feat, dtype='float32')
        data = data.reshape((-1, featsize + labelsize))
        label = data[:, FBANK_SIZE * numchs:]
        if self.numclasses == 2:
            label[label > 0.0] = 1.0
        else:
            label = np.round(label * LABEL_GAIN)
            label[label > self.numclasses - 1] = 0.0
        size1 = int(np.ceil(label.shape[0] / self.blockdec)) - self.blockcat + 1
        label1 = np.zeros((size1, LABEL_SIZE), dtype='float32')
        for tau in range(size1):
            label1[tau, :] = label[(tau + self.blockcat // 2) * self.blockdec, :]
        featall = np.zeros((size1, numchs, FBANK_SIZE * self.blockcat), dtype='float32')
        for n in range(numchs):
            feat = data[:, FBANK_SIZE * n:FBANK_SIZE * (n + 1)]
            for tau in range(size1):
                for i in range(self.blockcat):
                    featall[tau, n, FBANK_SIZE * i:FBANK_SIZE * (i + 1)] = feat[(tau + i) * self.blockdec, :]
        return (torch.from_numpy(featall), torch.from_numpy(label1).long())

    def get_sim(self, id):
        if False:
            while True:
                i = 10
        num_base = len(self.sims_base)
        if id < num_base:
            fs = self.sims_base[id]
        else:
            fs = self.sims_senior[id - num_base]
        return fs

class Worker(threading.Thread):
    """
    id:                 worker id
    dataset:            the dataset
    pool:               queue as the global data buffer
    """

    def __init__(self, id, dataset, pool):
        if False:
            return 10
        threading.Thread.__init__(self)
        self.id = id
        self.dataset = dataset
        self.pool = pool
        self.isrun = True
        self.nn = 0

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while self.isrun:
            self.nn += 1
            logger.debug(f'Worker {self.id:02d} running {self.nn:05d}:1')
            if self.isrun:
                data = self.dataset.getBatch(self.id)
            logger.debug(f'Worker {self.id:02d} running {self.nn:05d}:2')
            if self.isrun:
                self.pool.put(data)
            logger.debug(f'Worker {self.id:02d} running {self.nn:05d}:3')
        logger.info('KWSDataLoader: Worker {:02d} stopped.'.format(self.id))

    def stopWorker(self):
        if False:
            return 10
        '\n        stop the worker thread\n        '
        self.isrun = False

class KWSDataLoader:
    """ Load and organize audio data with multiple threads

    Args:
        dataset:            the dataset reference
        batchsize:          data batch size
        numworkers:         no. of workers
        prefetch:           prefetch factor
    """

    def __init__(self, dataset, batchsize, numworkers, prefetch=2):
        if False:
            while True:
                i = 10
        self.dataset = dataset
        self.batchsize = batchsize
        self.datamap = {}
        self.isrun = True
        self.pool = queue.Queue(numworkers * prefetch)
        self.workerlist = []
        for id in range(numworkers):
            w = Worker(id, dataset, self.pool)
            self.workerlist.append(w)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        while self.isrun:
            data = self.pool.get()
            self.pool.task_done()
            key = str(data[0].shape)
            batchl = self.datamap.get(key)
            if batchl is None:
                batchl = []
                self.datamap.update({key: batchl})
            batchl.append(data)
            if len(batchl) >= self.batchsize:
                featbatch = []
                labelbatch = []
                for (feat, label) in batchl:
                    featbatch.append(feat)
                    labelbatch.append(label)
                batchl.clear()
                feattensor = torch.stack(featbatch, dim=0)
                labeltensor = torch.stack(labelbatch, dim=0)
                if feattensor.shape[-2] == 1:
                    logger.debug('KWSDataLoader: Basetrain batch.')
                else:
                    logger.debug('KWSDataLoader: Finetune batch.')
                return (feattensor, labeltensor)
        return (None, None)

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        start multi-thread data loader\n        '
        for w in self.workerlist:
            w.start()

    def stop(self):
        if False:
            while True:
                i = 10
        '\n        stop data loader\n        '
        logger.info('KWSDataLoader: Stopping...')
        self.isrun = False
        for w in self.workerlist:
            w.stopWorker()
        while not self.pool.empty():
            self.pool.get(block=True, timeout=0.01)
        for w in self.workerlist:
            while not self.pool.empty():
                self.pool.get(block=True, timeout=0.01)
            w.join()
        logger.info('KWSDataLoader: All worker stopped.')