import shutil
import tempfile
from time import sleep, time
import pytest
from ding.data.model_loader import FileModelLoader
from ding.data.storage.file import FileModelStorage
from ding.model import DQN
from ding.config import compile_config
from dizoo.atari.config.serial.pong.pong_dqn_config import main_config, create_config
from os import path
import torch

@pytest.mark.tmp
def test_model_loader():
    if False:
        return 10
    tempdir = path.join(tempfile.gettempdir(), 'test_model_loader')
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    model = DQN(**cfg.policy.model)
    loader = FileModelLoader(model=model, dirname=tempdir, ttl=1)
    try:
        loader.start()
        model_storage = None

        def save_model(storage):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal model_storage
            model_storage = storage
        start = time()
        loader.save(save_model)
        save_time = time() - start
        print('Save time: {:.4f}s'.format(save_time))
        assert save_time < 0.1
        sleep(0.5)
        assert isinstance(model_storage, FileModelStorage)
        assert len(loader._files) > 0
        state_dict = loader.load(model_storage)
        model.load_state_dict(state_dict)
        sleep(2)
        assert not path.exists(model_storage.path)
        assert len(loader._files) == 0
    finally:
        if path.exists(tempdir):
            shutil.rmtree(tempdir)

@pytest.mark.benchmark
def test_model_loader_benchmark():
    if False:
        print('Hello World!')
    model = torch.nn.Sequential(torch.nn.Linear(1024, 1024), torch.nn.Linear(1024, 100))
    tempdir = path.join(tempfile.gettempdir(), 'test_model_loader')
    loader = FileModelLoader(model=model, dirname=tempdir)
    try:
        loader.start()
        count = 0

        def send_callback(_):
            if False:
                i = 10
                return i + 15
            nonlocal count
            count += 1
        start = time()
        for _ in range(5):
            loader.save(send_callback)
            sleep(0.2)
        while count < 5:
            sleep(0.001)
        assert time() - start < 1.2
    finally:
        if path.exists(tempdir):
            shutil.rmtree(tempdir)
        loader.shutdown()