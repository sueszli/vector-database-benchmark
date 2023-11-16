from ...ps.the_one_ps import TheOnePSRuntime
from ..runtime.collective_runtime import CollectiveRuntime
__all__ = []

class RuntimeFactory:

    def __init__(self):
        if False:
            return 10
        pass

    def _create_runtime(self, context):
        if False:
            while True:
                i = 10
        if 'use_fleet_ps' in context and context['use_fleet_ps']:
            ps_runtime = TheOnePSRuntime()
            ps_runtime._set_basic_info(context)
            return ps_runtime
        if context['role_maker']._is_collective:
            collective_runtime = CollectiveRuntime()
            collective_runtime._set_basic_info(context)
            return collective_runtime
        k_steps = context['valid_strategy'].a_sync_configs['k_steps']
        if not context['role_maker']._is_collective and k_steps >= 0:
            ps_runtime = TheOnePSRuntime()
            ps_runtime._set_basic_info(context)
            return ps_runtime