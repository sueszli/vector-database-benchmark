import typing
import transformers
from loguru import logger

class Printer(typing.Protocol):

    def __call__(self, value: int) -> None:
        if False:
            while True:
                i = 10
        ...

def _unpack(value):
    if False:
        i = 10
        return i + 15
    if len(value.shape) > 1 and value.shape[0] > 1:
        raise ValueError('HFStreamer only supports batch size 1')
    elif len(value.shape) > 1:
        value = value[0]
    return value.cpu().tolist()

class HFStreamer(transformers.generation.streamers.BaseStreamer):

    def __init__(self, input_ids, printer: Printer):
        if False:
            i = 10
            return i + 15
        self.input_ids = _unpack(input_ids)[::-1]
        self.printer = printer

    def put(self, value):
        if False:
            print('Hello World!')
        for token_id in _unpack(value):
            if self.input_ids:
                input_id = self.input_ids.pop()
                if input_id != token_id:
                    logger.warning(f'Input id {input_id} does not match output id {token_id}')
            else:
                self.printer(token_id)

    def end(self):
        if False:
            i = 10
            return i + 15
        pass