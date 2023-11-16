from PyQt5.QtCore import QMimeData
from feeluown.models import ModelType
model_mimetype_map = {ModelType.dummy.value: 'fuo-model/x-dummy', ModelType.song.value: 'fuo-model/x-song', ModelType.playlist.value: 'fuo-model/x-playlist', ModelType.album.value: 'fuo-model/x-album', ModelType.artist.value: 'fuo-model/x-artist', ModelType.lyric.value: 'fuo-model/x-lyric', ModelType.user.value: 'fuo-model/x-user'}

def get_model_mimetype(model):
    if False:
        while True:
            i = 10
    return model_mimetype_map[ModelType(model.meta.model_type).value]

class ModelMimeData(QMimeData):

    def __init__(self, model):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.model = model
        self._mimetype = get_model_mimetype(model)

    def setData(self, format, model):
        if False:
            while True:
                i = 10
        self._model = model

    def formats(self):
        if False:
            print('Hello World!')
        return [self._mimetype]

    def hasFormat(self, format):
        if False:
            while True:
                i = 10
        return format == self._mimetype

    def data(self, format):
        if False:
            i = 10
            return i + 15
        if format == self._mimetype:
            return self.model
        return None