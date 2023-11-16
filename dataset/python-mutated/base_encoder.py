import deeplake
from abc import ABC
from typing import Any, Sequence, Optional
from deeplake.constants import ENCODING_DTYPE
import numpy as np
LAST_SEEN_INDEX_COLUMN = -1

class Encoder(ABC):
    last_row = 0

    def is_index_in_last_row(self, arr, index) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if `index` is in the self.last_row of of encoder.'
        row = self.last_row
        return arr[row, -1] >= index and (row == 0 or arr[row - 1, -1] < index)

    def check_last_row(self, global_sample_index: int):
        if False:
            for i in range(10):
                print('nop')
        'Takes a look at self.last_row and tries to find chunk id without binary search by looking at the current and next row.'
        arr = self._encoded
        if self.last_row < len(arr) and self.is_index_in_last_row(arr, global_sample_index):
            return self.last_row
        elif self.last_row < len(arr) - 1:
            self.last_row += 1
            if self.is_index_in_last_row(arr, global_sample_index):
                return self.last_row
        return None

    def __init__(self, encoded=None, dtype=ENCODING_DTYPE):
        if False:
            return 10
        'Base class for custom encoders that allow reading meta information from sample indices without decoding the entire encoded state.\n\n        Handles heavy lifting logic for:\n            - Chunk ID encoder\n            - Shape encoder\n            - Byte positions encoder\n\n        Lookup algorithm is essentially the same for all encoders, however the details are different.\n        You can find all of this information in their respective classes.\n\n        Layout:\n            `_encoded` is a 2D array.\n\n            Best case scenario:\n                The best case scenario is when all samples have the same meta and can thus be stored in a single row,\n                providing a O(1) lookup.\n\n            Worst case scenario:\n                The worst case scenario is when every sample has different meta values. This means the number of rows is equal to the number\n                of samples, providing a O(log(N)) lookup.\n\n            Lookup algorithm:\n                To get the decoded meta for some sample index, you do a binary search over the column `LAST_SEEN_INDEX_COLUMN`. This will give you\n                the row that corresponds to that sample index (since the right-most column is our "last index" for that meta information).\n                Then, you decode the row and that is your meta!\n\n        Raises:\n            ValueError: If `encoded` is not the right dtype.\n\n        Args:\n            encoded (np.ndarray): Encoded state, if None state is empty. Helpful for deserialization. Defaults to None.\n            dtype (np.dtype): Dtype of the encoder. Defaults to `ENCODING_DTYPE`.\n        '
        if not hasattr(self, '_num_columns'):
            self._num_columns = 2
        self.dtype = dtype
        if isinstance(encoded, list):
            encoded = np.array(encoded, dtype=self.dtype)
        if encoded is None:
            encoded = np.zeros((0, self._num_columns), dtype=self.dtype)
        self._encoded = encoded
        if self._encoded.dtype != self.dtype:
            raise ValueError(f'Encoding dtype should be {self.dtype}, instead got {self._encoded.dtype}')
        self.version = deeplake.__version__
        self.is_dirty = True

    @property
    def array(self):
        if False:
            for i in range(10):
                print('nop')
        return self._encoded

    @property
    def nbytes(self):
        if False:
            return 10
        return self.array.nbytes

    @property
    def num_samples(self) -> int:
        if False:
            print('Hello World!')
        if len(self._encoded) == 0:
            return 0
        return int(self._encoded[-1, LAST_SEEN_INDEX_COLUMN] + 1)

    def num_samples_at(self, row_index: int) -> int:
        if False:
            return 10
        "Calculates the number of samples a row in the encoding corresponds to.\n\n        Args:\n            row_index (int): This index will be used when indexing `self._encoded`.\n\n        Returns:\n            int: Representing the number of samples that a row's derivable value represents.\n        "
        lower_bound = 0
        if len(self._encoded) > 1 and row_index > 0:
            lower_bound = self._encoded[row_index - 1, LAST_SEEN_INDEX_COLUMN] + 1
        upper_bound = self._encoded[row_index, LAST_SEEN_INDEX_COLUMN] + 1
        return int(upper_bound - lower_bound)

    def translate_index(self, local_sample_index: int) -> int:
        if False:
            i = 10
            return i + 15
        'Searches for the row index for where `local_sample_index` exists within `self._encoded`.\n        This method is worst case log(N) due to the binary search.\n\n        Args:\n            local_sample_index (int): Index representing a sample. Localized to `self._encoded`.\n\n        Raises:\n            IndexError: Cannot index when there are no samples to index into.\n\n        Returns:\n            int: The index of the corresponding row inside the encoded state.\n        '
        if len(self._encoded) == 0:
            raise IndexError(f'Index {local_sample_index} is out of bounds for an empty encoder.')
        if local_sample_index < 0:
            local_sample_index += self.num_samples
        row_index = self.check_last_row(local_sample_index)
        if row_index is None:
            row_index = np.searchsorted(self._encoded[:, LAST_SEEN_INDEX_COLUMN], local_sample_index)
            self.last_row = row_index
        return row_index

    def register_samples(self, item: Any, num_samples: int, row: Optional[int]=None):
        if False:
            print('Hello World!')
        "Register `num_samples` as `item`. Combines when the `self._combine_condition` returns True.\n        This method adds data to `self._encoded` without decoding.\n\n        Args:\n            item (Any): General input, will be passed along to subclass methods.\n            num_samples (int): Number of samples that have `item`'s value. Will be passed along to subclass methods.\n            row (Optional[int]): Parameter that shows to which chunk the samples need to be added\n        "
        self._validate_incoming_item(item, num_samples)
        if self.num_samples != 0:
            if self._combine_condition(item):
                last_index = self._encoded[-1, LAST_SEEN_INDEX_COLUMN]
                if row is not None:
                    self._encoded[row][1] += num_samples
                else:
                    new_last_index = self._derive_next_last_index(last_index, num_samples)
                    self._encoded[-1, LAST_SEEN_INDEX_COLUMN] = new_last_index
            else:
                decomposable = self._make_decomposable(item)
                last_index = self._encoded[-1, LAST_SEEN_INDEX_COLUMN]
                next_last_index = self._derive_next_last_index(last_index, num_samples)
                if row is not None:
                    self._encoded[:, LAST_SEEN_INDEX_COLUMN] += num_samples
                    shape_entry = np.array([*decomposable, num_samples - 1], dtype=self.dtype)
                    self._encoded = np.insert(self._encoded, row, shape_entry, axis=0)
                else:
                    shape_entry = np.array([[*decomposable, next_last_index]], dtype=self.dtype)
                    self._encoded = np.concatenate([self._encoded, shape_entry], axis=0)
        else:
            decomposable = self._make_decomposable(item)
            self._encoded = np.array([[*decomposable, num_samples - 1]], dtype=self.dtype)
        self.is_dirty = True

    def _validate_incoming_item(self, item: Any, num_samples: int):
        if False:
            for i in range(10):
                print('nop')
        "Raises appropriate exceptions for when `item` or `num_samples` are invalid.\n        Subclasses should override this method when applicable.\n\n        Args:\n            item (Any): General input, will be passed along to subclass methods.\n            num_samples (int): Number of samples that have `item`'s value. Will be passed along to subclass methods.\n\n        Raises:\n            ValueError: For the general case, `num_samples` should be > 0.\n        "
        if num_samples <= 0:
            raise ValueError(f'`num_samples` should be > 0. Got: {num_samples}')

    def _combine_condition(self, item: Any, compare_row_index: int=-1):
        if False:
            return 10
        'Should determine if `item` can be combined with a row in `self._encoded`.'
        return False

    def _derive_next_last_index(self, last_index, num_samples: int):
        if False:
            for i in range(10):
                print('nop')
        'Calculates what the next last index should be.'
        return last_index + num_samples

    def _make_decomposable(self, item: Any, compare_row_index: int=-1) -> Sequence:
        if False:
            return 10
        'Should return a value that can be decompsed with the `*` operator. Example: `*(1, 2)`'
        return item

    def _derive_value(self, row: np.ndarray, row_index: int, local_sample_index: int):
        if False:
            for i in range(10):
                print('nop')
        'Given a row of `self._encoded`, this method should implement how `__getitem__` hands a value to the caller.'
        return np.ndarray([])

    def __getitem__(self, local_sample_index: int, return_row_index: bool=False) -> Any:
        if False:
            while True:
                i = 10
        'Derives the value at `local_sample_index`.\n\n        Args:\n            local_sample_index (int): Index of the sample for the desired value.\n            return_row_index (bool): If True, the index of the row that the value was derived from is returned as well.\n                Defaults to False.\n\n        Returns:\n            Any: Either just a singular derived value, or a tuple with the derived value and the row index respectively.\n        '
        row_index = self.translate_index(local_sample_index)
        value = self._derive_value(self._encoded[row_index], row_index, local_sample_index)
        if return_row_index:
            return (value, row_index)
        return value

    def __setitem__(self, local_sample_index: int, item: Any):
        if False:
            for i in range(10):
                print('nop')
        'Update the encoded value at a given index. Depending on the state, this may increase/decrease\n        the size of the state.\n\n        Updating:\n            Updation is executed by going through a list of possible actions and trying to reduce the cost delta.\n\n            Cost:\n                Cost is defined as `len(self._encoded)`.\n                The "cost delta" is the number of rows added/removed from `self._encoded` as a result of the action.\n\n            Actions are chosen assuming `self._encoded` is already encoded properly.\n\n            Note:\n                An action that is "upwards" is being performed towards idx=0\n                An action that is "downwards" is being performed away from idx=0\n\n            Actions in order of execution:\n                no change    (cost delta = 0)\n                squeeze      (cost delta = -2)\n                squeeze up   (cost delta = -1)\n                squeeze down (cost delta = -1)\n                move up      (cost delta = 0)\n                move down    (cost delta = 0)\n                replace      (cost delta = 0)\n                split up     (cost delta = +1)\n                split down   (cost delta = +1)\n                split middle (cost delta = +2)\n\n        Args:\n            local_sample_index (int): Index representing a sample. Localized to `self._encoded`.\n            item (Any): Item compatible with the encoder subclass.\n\n        Raises:\n            ValueError: If no update actions were taken.\n        '
        row_index = self.translate_index(local_sample_index)
        actions = (self._try_not_changing, self._setup_update, self._try_squeezing, self._try_squeezing_up, self._try_squeezing_down, self._try_moving_up, self._try_moving_down, self._try_replacing, self._try_splitting_up, self._try_splitting_down, self._try_splitting_middle)
        action_taken = None
        for action in actions:
            if action(item, row_index, local_sample_index):
                action_taken = action
                break
        if action_taken is None:
            raise ValueError(f'Update could not be executed for idx={local_sample_index}, item={str(item)}')
        self._post_process_state(start_row_index=max(row_index - 2, 0))
        self._reset_update_state()
        self.is_dirty = True

    def _post_process_state(self, start_row_index: int):
        if False:
            while True:
                i = 10
        'Overridden when more complex columns exist in subclasses. Example: byte positions.'
        pass

    def _reset_update_state(self):
        if False:
            return 10
        self._has_above = None
        self._has_below = None
        self._can_combine_above = None
        self._can_combine_below = None
        self._decomposable_item = None
        self._num_samples_at_row = None

    def _setup_update(self, item: Any, row_index: int, local_sample_index: int):
        if False:
            for i in range(10):
                print('nop')
        'Setup the state variables for preceeding actions. Used for updating.'
        self._has_above = row_index > 0
        self._has_below = row_index + 1 < len(self._encoded)
        self._can_combine_above = self._has_above and self._encoded[row_index - 1][LAST_SEEN_INDEX_COLUMN] + 1 == local_sample_index and self._combine_condition(item, row_index - 1)
        self._can_combine_below = self._has_below and self._encoded[row_index][LAST_SEEN_INDEX_COLUMN] == local_sample_index and self._combine_condition(item, row_index + 1)
        self._decomposable_item = self._make_decomposable(item, row_index)
        self._num_samples_at_row = self.num_samples_at(row_index)

    def _try_not_changing(self, item: Any, row_index: int, *_) -> bool:
        if False:
            return 10
        'If `item` already is the value at `row_index`, no need to make any updates.\n\n        Cost delta = 0\n\n        Example:\n            Start:\n                item    last index\n                ------------------\n                A       10\n                B       11\n\n            Update:\n                self[5] = A\n\n            End:\n                item    last index\n                ------------------\n                A       10\n                B       11\n        '
        return self._combine_condition(item, row_index)

    def _try_squeezing(self, item: Any, row_index: int, *_) -> bool:
        if False:
            print('Hello World!')
        'If update results in the above and below rows in `self._encoded`\n        to match the incoming item, just combine them all into a single row.\n\n        Cost delta = -2\n\n        Example:\n            Start:\n                item    last index\n                ------------------\n                A       10\n                B       11\n                A       15\n\n            Update:\n                self[11] = A\n\n            End:\n                item    last index\n                ------------------\n                A       15\n        '
        if self._num_samples_at_row != 1:
            return False
        if not (self._has_above and self._has_below):
            return False
        if not (self._can_combine_above and self._can_combine_below):
            return False
        start = self._encoded[:row_index - 1]
        end = self._encoded[row_index + 1:]
        self._encoded = np.concatenate((start, end))
        return True

    def _try_squeezing_up(self, item: Any, row_index: int, *_) -> bool:
        if False:
            print('Hello World!')
        'If update results in the above row in `self._encoded`\n        matching the incoming item, just combine them into a single row.\n\n        Cost delta = -1\n\n        Example:\n            Start:\n                item    last index\n                ------------------\n                A       10\n                B       11\n                C       15\n\n            Update:\n                self[11] = A\n\n            End:\n                item    last index\n                ------------------\n                A       11\n                C       15\n        '
        if self._num_samples_at_row != 1:
            return False
        if not self._has_above:
            return False
        if not self._can_combine_above:
            return False
        start = self._encoded[:row_index]
        end = self._encoded[row_index + 1:]
        start[-1, LAST_SEEN_INDEX_COLUMN] += 1
        self._encoded = np.concatenate((start, end))
        return True

    def _try_squeezing_down(self, item: Any, row_index: int, *_) -> bool:
        if False:
            print('Hello World!')
        'If update results in the below row in `self._encoded`\n        matching the incoming item, just combine them into a single row.\n\n        Cost delta = -1\n\n        Example:\n            Start:\n                item    last index\n                ------------------\n                A       10\n                B       11\n                C       15\n\n            Update:\n                self[11] = C\n\n            End:\n                item    last index\n                ------------------\n                A       10\n                C       15\n        '
        if self._num_samples_at_row != 1:
            return False
        if not self._has_below:
            return False
        if not self._can_combine_below:
            return False
        start = self._encoded[:row_index]
        end = self._encoded[row_index + 1:]
        self._encoded = np.concatenate((start, end))
        return True

    def _try_moving_up(self, item: Any, row_index: int, *_) -> bool:
        if False:
            while True:
                i = 10
        'If `item` exists in the row above `row_index`, then we can just use the above row to encode `item`.\n\n        Cost delta = 0\n\n        Example:\n            Start:\n                item    last index\n                ------------------\n                A       10\n                B       15\n\n            Update:\n                self[11] = A\n\n            End:\n                item    last index\n                ------------------\n                A       11\n                B       15\n        '
        if self._can_combine_below or not self._can_combine_above:
            return False
        self._encoded[row_index - 1, LAST_SEEN_INDEX_COLUMN] += 1
        return True

    def _try_moving_down(self, item: Any, row_index: int, *_) -> bool:
        if False:
            return 10
        'If `item` exists in the row below `row_index`, then we can just use the below row to encode `item`.\n\n        Cost delta = 0\n\n        Example:\n            Start:\n                item    last index\n                ------------------\n                A       10\n                B       15\n\n            Update:\n                self[10] = B\n\n            End:\n                item    last index\n                ------------------\n                A       9\n                B       15\n        '
        if self._can_combine_above or not self._can_combine_below:
            return False
        self._encoded[row_index, LAST_SEEN_INDEX_COLUMN] -= 1
        return True

    def _try_replacing(self, item: Any, row_index: int, *args) -> bool:
        if False:
            while True:
                i = 10
        'If the value encoded at `row_index` only exists for a single index, then `row_index`\n        can be directly replaced with `item`.\n\n        Cost delta = 0\n\n        Example:\n            Start:\n                item    last index\n                ------------------\n                A       10\n                B       11\n                C       20\n\n            Update:\n                self[11] = D\n\n            End:\n                item    last index\n                ------------------\n                A       10\n                D       11\n                C       20\n        '
        if self._num_samples_at_row != 1:
            return False
        self._encoded[row_index, :LAST_SEEN_INDEX_COLUMN] = item
        return True

    def _try_splitting_up(self, item: Any, row_index: int, local_sample_index: int) -> bool:
        if False:
            return 10
        "If the row at `row_index` is being updated on the first index it is responsible for,\n        AND the above row doesn't match `item`, a new row needs to be created above.\n\n        Cost delta = +1\n\n        Example:\n            Start:\n                item    last index\n                ------------------\n                A       10\n                B       15\n                C       20\n\n            Update:\n                self[11] = D\n\n            End:\n                item    last index\n                ------------------\n                A       10\n                D       11\n                B       15\n                C       20\n        "
        above_last_index = -1
        if self._has_above:
            above_last_index = self._encoded[row_index - 1, LAST_SEEN_INDEX_COLUMN]
        if above_last_index != local_sample_index - 1:
            return False
        start = self._encoded[:max(0, row_index)]
        end = self._encoded[row_index:]
        new_row = np.array([*self._decomposable_item, local_sample_index], dtype=self.dtype)
        self._encoded = np.concatenate((start, [new_row], end))
        return True

    def _try_splitting_down(self, item: Any, row_index: int, local_sample_index: int) -> bool:
        if False:
            print('Hello World!')
        "If the row at `row_index` is being updated on the last index it is responsible for,\n        AND the below row doesn't match `item`, a new row needs to be created below.\n\n        Cost delta = +1\n\n        Example:\n            Start:\n                item    last index\n                ------------------\n                A       10\n                B       15\n                C       20\n\n            Update:\n                self[15] = D\n\n            End:\n                item    last index\n                ------------------\n                A       10\n                B       14\n                D       15\n                C       20\n        "
        last_index = self._encoded[row_index, LAST_SEEN_INDEX_COLUMN]
        if last_index != local_sample_index:
            return False
        start = self._encoded[:row_index + 1]
        end = self._encoded[row_index + 1:]
        start[-1, LAST_SEEN_INDEX_COLUMN] -= 1
        new_row = np.array([*self._decomposable_item, local_sample_index], dtype=self.dtype)
        self._encoded = np.concatenate((start, [new_row], end))
        return True

    def _try_splitting_middle(self, item: Any, row_index: int, local_sample_index: int) -> bool:
        if False:
            i = 10
            return i + 15
        'If the row at `row_index` is being updated on an index in the middle of the samples it is responsible for,\n        a new row needs to be created above AND below.\n\n        Cost delta = +2\n\n        Example:\n            Start:\n                item    last index\n                ------------------\n                A       10\n                B       15\n                A       20\n\n            Update:\n                self[13] = A\n\n            End:\n                item    last index\n                ------------------\n                A       10\n                B       12\n                A       13\n                B       15\n                A       20\n        '
        start = np.array(self._encoded[:row_index + 1])
        new_row = np.array([*self._decomposable_item, local_sample_index], dtype=self.dtype)
        end = self._encoded[row_index:]
        start[-1, LAST_SEEN_INDEX_COLUMN] = local_sample_index - 1
        self._encoded = np.concatenate((start, [new_row], end))
        return True

    def _num_samples_in_last_row(self):
        if False:
            return 10
        if len(self._encoded) == 0:
            return 0
        elif len(self._encoded) == 1:
            return self._encoded[-1][LAST_SEEN_INDEX_COLUMN] + 1
        else:
            return self._encoded[-1][LAST_SEEN_INDEX_COLUMN] - self._encoded[-2][LAST_SEEN_INDEX_COLUMN]

    def pop(self, index: Optional[int]=None):
        if False:
            while True:
                i = 10
        if index is None:
            index = self.get_last_index_for_pop()
        (_, row) = self.__getitem__(index, return_row_index=True)
        prev = -1 if row == 0 else self._encoded[row - 1, LAST_SEEN_INDEX_COLUMN]
        num_samples_in_row = self._encoded[row, LAST_SEEN_INDEX_COLUMN] - prev
        if num_samples_in_row == 0:
            raise ValueError('No samples to pop')
        self._encoded[row:, LAST_SEEN_INDEX_COLUMN] -= 1
        if num_samples_in_row == 1:
            self._encoded = np.delete(self._encoded, row, axis=0)
        self.is_dirty = True

    def is_empty(self) -> bool:
        if False:
            while True:
                i = 10
        return len(self._encoded) == 0

    def tobytes(self) -> memoryview:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @classmethod
    def frombuffer(cls, buffer: bytes):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def get_last_index_for_pop(self) -> int:
        if False:
            i = 10
            return i + 15
        num_samples = self.num_samples
        if num_samples == 0:
            raise ValueError('No samples to pop')
        return num_samples - 1