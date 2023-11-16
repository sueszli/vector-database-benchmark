from ..utils.hybrid_parallel_util import broadcast_dp_parameters, broadcast_sep_parameters, broadcast_sharding_parameters
from ..utils.log_util import logger
from .meta_parallel_base import MetaParallelBase
__all__ = []

class SegmentParallel(MetaParallelBase):

    def __init__(self, layers, hcg, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(layers, hcg, **kwargs)

    def _prepare_for_model(self):
        if False:
            i = 10
            return i + 15
        logger.info('start broadcast sep parameters')
        broadcast_sep_parameters(self._layers, self._hcg)
        if self._hcg.get_sharding_parallel_world_size() > 1:
            logger.info('start broadcast sharding parameters')
            broadcast_sharding_parameters(self._layers, self._hcg)
        if self._hcg.get_data_parallel_world_size() > 1:
            logger.info('start broadcast dp parameters')
            broadcast_dp_parameters(self._layers, self._hcg)