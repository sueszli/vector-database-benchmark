from typing import List, Iterable, Sequence, Optional
from allennlp.common.registrable import Registrable
from allennlp.data.instance import Instance

class BatchSampler(Registrable):

    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def get_batch_size(self) -> Optional[int]:
        if False:
            print('Hello World!')
        '\n        Not all `BatchSamplers` define a consistent `batch_size`, but those that\n        do should override this method.\n        '
        return None