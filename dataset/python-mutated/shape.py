from deeplake.core.meta.encode.base_encoder import Encoder, LAST_SEEN_INDEX_COLUMN
from deeplake.constants import ENCODING_DTYPE
from typing import Tuple
from deeplake.core.storage.provider import StorageProvider
import numpy as np

class ShapeEncoder(Encoder):

    def _derive_value(self, row: np.ndarray, *_) -> Tuple:
        if False:
            while True:
                i = 10
        return tuple(row[:LAST_SEEN_INDEX_COLUMN])

    @property
    def dimensionality(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self[0])

    def _combine_condition(self, shape: Tuple[int], compare_row_index: int=-1) -> bool:
        if False:
            print('Hello World!')
        last_shape = self._derive_value(self._encoded[compare_row_index])
        return shape == last_shape