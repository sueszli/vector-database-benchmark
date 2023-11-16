from base64 import b64decode, b64encode
import io
from multiprocessing import Process
import time
from typing import List
import unittest
from uuid import uuid4
import numpy as np
import pandas as pd
import os
from bigdl.ppml.fl import *
from torch import Tensor, nn
import torch
from bigdl.ppml.fl.utils import FLTest
import shutil
from bigdl.dllib.utils.encryption_utils import *
resource_path = os.path.join(os.path.dirname(__file__), '../../resources')

class TestJitEncrypt(FLTest):
    fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    model_path = '/tmp/vfl_jit_model'

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        if not os.path.exists(f'{TestJitEncrypt.model_path}'):
            os.makedirs(f'{TestJitEncrypt.model_path}', exist_ok=True)

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        if os.path.exists(TestJitEncrypt.model_path):
            shutil.rmtree(TestJitEncrypt.model_path)

    def test_encrypt_to_file_and_decrypt(self) -> None:
        if False:
            while True:
                i = 10
        model = NeuralNetwork()
        m = torch.jit.script(model)
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        key = str(uuid4())
        salt = str(uuid4())
        byte_buffer = buffer.getvalue()
        encrypted = encrypt_bytes_with_AES_CBC(byte_buffer, key, salt)
        with open(f'{TestJitEncrypt.model_path}/model.weights', 'wb') as f:
            f.write(encrypted)
            logging.info(f'Writing model to {TestJitEncrypt.model_path}/model.weights')
        with open(f'{TestJitEncrypt.model_path}/model.weights', 'rb') as f:
            encrypted_read = f.read()
            logging.info(f'Loading model to {TestJitEncrypt.model_path}/model.weights')
        loaded = decrypt_bytes_with_AES_CBC(encrypted_read, key, salt)
        buffer_load = io.BytesIO(loaded)
        loaded_model = torch.jit.load(buffer_load)
        self.assertEqual(len(list(model.parameters())), len(list(loaded_model.parameters())))

class NeuralNetwork(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.sequential_1 = nn.Sequential(nn.Linear(28 * 28, 512), nn.ReLU())
        self.sequential_2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.flatten(x)
        x = self.sequential_1(x)
        x = self.sequential_2(x)
        return x
if __name__ == '__main__':
    unittest.main()