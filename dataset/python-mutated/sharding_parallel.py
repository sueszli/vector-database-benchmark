from ..utils.hybrid_parallel_util import broadcast_dp_parameters, broadcast_sharding_parameters
from ..utils.log_util import logger
from .meta_parallel_base import MetaParallelBase
__all__ = []

class ShardingParallel(MetaParallelBase):

    def __init__(self, layers, hcg, **kwargs):
        if False:
            return 10
        super().__init__(layers, hcg, **kwargs)

    def _prepare_for_model(self):
        if False:
            for i in range(10):
                print('nop')
        logger.info('start broadcast sharding parameters')
        broadcast_sharding_parameters(self._layers, self._hcg)
        if self._hcg.get_data_parallel_world_size() > 1:
            logger.info('start broadcast dp parameters')
            broadcast_dp_parameters(self._layers, self._hcg)
        logger.info("sharding's parameters is ready")