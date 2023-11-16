import uuid
from ray.data.preprocessor import Preprocessor

class DummyPreprocessor(Preprocessor):
    _is_fittable = False

    def __init__(self, transform=lambda b: b):
        if False:
            i = 10
            return i + 15
        self.id = uuid.uuid4()
        self.transform = transform

    def transform_batch(self, batch):
        if False:
            while True:
                i = 10
        self._batch_transformed = True
        return self.transform(batch)

    def _transform_pandas(self, df):
        if False:
            i = 10
            return i + 15
        return df

    @property
    def has_preprocessed(self):
        if False:
            i = 10
            return i + 15
        return hasattr(self, '_batch_transformed')

    def __eq__(self, other_preprocessor):
        if False:
            for i in range(10):
                print('nop')
        return self.id == other_preprocessor.id