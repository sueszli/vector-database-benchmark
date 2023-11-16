import os
import io
import shutil
import tempfile
from abc import ABCMeta, abstractmethod
import torch
from bigdl.dllib.utils.log4Error import invalidInputError

class ModelIO(metaclass=ABCMeta):

    @abstractmethod
    def get_state_dict(self):
        if False:
            i = 10
            return i + 15
        'Returns the state of the runner.'
        pass

    @abstractmethod
    def load_state_dict(self, state):
        if False:
            return 10
        'Sets the state of the model.'
        pass

    @abstractmethod
    def save_checkpoint(self, filepath, save_weights_only=False):
        if False:
            for i in range(10):
                print('nop')
        'Save checkpoint.'
        pass

    @abstractmethod
    def remove_checkpoint(self, filepath):
        if False:
            while True:
                i = 10
        'Remove checkpoint'
        pass

    def get_state_stream(self):
        if False:
            print('Hello World!')
        'Returns a bytes object for the state dict.'
        state_dict = self.get_state_dict()
        state_stream = ModelIO._state_dict2stream(state_dict)
        return state_stream

    def load_state_stream(self, byte_obj):
        if False:
            print('Hello World!')
        'Loads a bytes object the training state dict.'
        state_dict = ModelIO._state_stream2dict(byte_obj)
        return self.load_state_dict(state_dict)

    def load_checkpoint(self, filepath):
        if False:
            while True:
                i = 10
        from bigdl.orca.data.file import get_remote_file_to_local
        file_name = os.path.basename(filepath)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file_name)
        try:
            get_remote_file_to_local(filepath, temp_path)
            state_dict = torch.load(temp_path)
        finally:
            shutil.rmtree(temp_dir)
        self.load_state_dict(state_dict)

    @staticmethod
    def _state_dict2stream(state_dict):
        if False:
            print('Hello World!')
        _buffer = io.BytesIO()
        torch.save(state_dict, _buffer)
        return _buffer.getvalue()

    @staticmethod
    def _state_stream2dict(byte_obj):
        if False:
            i = 10
            return i + 15
        _buffer = io.BytesIO(byte_obj)
        state_dict = torch.load(_buffer)
        return state_dict