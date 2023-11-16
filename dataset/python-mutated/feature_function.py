import numpy as np
import os
import pickle
from common.util.log_helper import LogHelper

class Features:

    def __init__(self, model_name, features=list(), label_name='label', base_path='features'):
        if False:
            return 10
        self.feature_functions = features
        self.vocabs = dict()
        self.label_name = label_name
        self.base_path = base_path
        self.logger = LogHelper.get_logger(Features.__name__)
        self.mname = model_name

    def check_needs_generate(self, train, dev, test):
        if False:
            for i in range(10):
                print('nop')
        for ff in self.feature_functions:
            ffpath = os.path.join(self.base_path, ff.get_name())
            if not os.path.exists(ffpath):
                os.makedirs(ffpath)
            if not os.path.exists(os.path.join(ffpath, 'train')) or (dev is not None and (not os.path.exists(os.path.join(ffpath, 'dev')))) or (test is not None and (not os.path.exists(os.path.join(ffpath, 'test')))) or (os.getenv('GENERATE', '').lower() in ['y', '1', 't', 'yes']):
                return True
        return False

    def load(self, train, dev=None, test=None):
        if False:
            return 10
        train_fs = []
        dev_fs = []
        test_fs = []
        if self.check_needs_generate(train, dev, test):
            self.inform(train, dev, test)
        else:
            try:
                self.load_vocab(self.mname)
            except:
                self.logger.info('Could not load vocab. Regenerating')
                self.inform(train, dev, test)
        for ff in self.feature_functions:
            train_fs.append(self.generate_or_load(ff, train, 'train'))
            dev_fs.append(self.generate_or_load(ff, dev, 'dev'))
            test_fs.append(self.generate_or_load(ff, test, 'test'))
        self.save_vocab(self.mname)
        return (self.out(train_fs, train), self.out(dev_fs, dev), self.out(test_fs, test))

    def out(self, features, ds):
        if False:
            i = 10
            return i + 15
        if ds is not None:
            return (np.hstack(features) if len(features) > 1 else features[0], self.labels(ds.data))
        return ([[]], [])

    def generate_or_load(self, feature, dataset, name):
        if False:
            while True:
                i = 10
        ffpath = os.path.join(self.base_path, feature.get_name())
        if dataset is not None:
            if os.path.exists(os.path.join(ffpath, name)) and os.getenv('GENERATE', '').lower() not in ['y', '1', 't', 'yes']:
                self.logger.info('Loading Features for {0}.{1}'.format(feature, name))
                with open(os.path.join(ffpath, name), 'rb') as f:
                    features = pickle.load(f)
            else:
                self.logger.info('Generating Features for {0}.{1}'.format(feature, name))
                features = feature.lookup(dataset.data)
                with open(os.path.join(ffpath, name), 'wb+') as f:
                    pickle.dump(features, f)
            return features
        return None

    def lookup(self, dataset):
        if False:
            for i in range(10):
                print('nop')
        fs = []
        for feature_function in self.feature_functions:
            print('Load {0}'.format(feature_function))
            fs.append(feature_function.lookup(dataset.data))
        return self.out(fs, dataset)

    def labels(self, data):
        if False:
            while True:
                i = 10
        return [datum[self.label_name] for datum in data]

    def inform(self, train, dev=None, test=None):
        if False:
            return 10
        for feature_function in self.feature_functions:
            self.logger.info('Inform {0} with {1} data'.format(feature_function, len(train.data)))
            feature_function.inform(train.data, dev.data if dev is not None else None, test.data if test is not None else None)

    def save_vocab(self, mname):
        if False:
            for i in range(10):
                print('nop')
        for ff in self.feature_functions:
            ff.save(mname)

    def load_vocab(self, mname):
        if False:
            print('Hello World!')
        for ff in self.feature_functions:
            ff.load(mname)

class FeatureFunction:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def inform(self, train, dev, test):
        if False:
            print('Hello World!')
        raise NotImplementedError('Not Implemented Here')

    def lookup(self, data):
        if False:
            print('Hello World!')
        return self.process(data)

    def process(self, data):
        if False:
            return 10
        pass

    def load_vocab(self, mname):
        if False:
            print('Hello World!')
        pass

    def save_vocab(self, mname):
        if False:
            i = 10
            return i + 15
        pass