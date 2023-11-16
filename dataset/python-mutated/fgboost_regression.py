from time import time
from bigdl.dllib.utils.common import JavaValue
from bigdl.ppml.fl.data_utils import *
from bigdl.ppml.fl import *
from bigdl.ppml.fl.fgboost.utils import add_data
import logging

class FGBoostRegression(FLClientClosable):

    def __init__(self, jvalue=None, learning_rate: float=0.1, max_depth=7, min_child_size=1, server_model_path=None):
        if False:
            while True:
                i = 10
        self.bigdl_type = 'float'
        super().__init__(jvalue, self.bigdl_type, learning_rate, max_depth, min_child_size, server_model_path)

    def fit(self, x, y=None, num_round=5, **kargs):
        if False:
            while True:
                i = 10
        x = convert_to_numpy(x)
        y = convert_to_numpy(y) if y is not None else None
        add_data(x, self.value, 'fgBoostFitAdd', self.bigdl_type)
        ts = time()
        (x, y) = convert_to_jtensor(x, y, **kargs)
        te = time()
        logging.info(f'ndarray to jtensor: [{te - ts} s]')
        return callBigDlFunc(self.bigdl_type, 'fgBoostFitCall', self.value, y, num_round)

    def evaluate(self, x, y=None, batchsize=4, **kargs):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = convert_to_jtensor(x, y, **kargs)
        return callBigDlFunc(self.bigdl_type, 'fgBoostEvaluate', self.value, x, y)

    def predict(self, x, batchsize=4, **kargs):
        if False:
            for i in range(10):
                print('nop')
        i = 0
        result = []
        while i + batchsize < len(x):
            x_batch = x[i:i + batchsize]
            (x_batch, _) = convert_to_jtensor(x_batch, **kargs)
            result_batch = callBigDlFunc(self.bigdl_type, 'fgBoostPredict', self.value, x_batch).to_ndarray()
            result.append(result_batch.flatten())
            i += batchsize
        x_batch = x[i:]
        (x_batch, _) = convert_to_jtensor(x_batch, **kargs)
        result_batch = callBigDlFunc(self.bigdl_type, 'fgBoostPredict', self.value, x_batch).to_ndarray()
        result.append(result_batch.flatten())
        flat_result = [x for xs in result for x in xs]
        return np.array(flat_result)

    def save_model(self, dest):
        if False:
            for i in range(10):
                print('nop')
        callBigDlFunc(self.bigdl_type, 'fgBoostRegressionSave', self.value, dest)

    @classmethod
    def load_model(cls, src):
        if False:
            while True:
                i = 10
        return cls(jvalue=callBigDlFunc('float', 'fgBoostRegressionLoad', src))

    def load_server_model(self, model_path):
        if False:
            print('Hello World!')
        callBigDlFunc(self.bigdl_type, 'fgBoostLoadServerModel', self.value, model_path)