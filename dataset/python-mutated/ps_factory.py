from .ps_program_builder import *
from .public import *
__all__ = ['PsProgramBuilder', 'GeoPsProgramBuilder', 'CpuSyncPsProgramBuilder', 'CpuAsyncPsProgramBuilder', 'GpuPsProgramBuilder', 'HeterAsyncPsProgramBuilder', 'FlPsProgramBuilder', 'NuPsProgramBuilder']

class PsProgramBuilderFactory:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def _create_ps_program_builder(self, pass_ctx):
        if False:
            print('Hello World!')
        attrs = pass_ctx._attrs
        if attrs['ps_mode'] == DistributedMode.GEO:
            if len(attrs['local_sparse']) != 0:
                return globals()['NuPsProgramBuilder'](pass_ctx)
            else:
                return globals()['GeoPsProgramBuilder'](pass_ctx)
        elif attrs['use_ps_gpu']:
            return globals()['GpuPsProgramBuilder'](pass_ctx)
        elif attrs['is_heter_ps_mode'] and (not attrs['is_fl_ps_mode']):
            return globals()['HeterAsyncPsProgramBuilder'](pass_ctx)
        elif 'is_fl_ps_mode' in attrs and attrs['is_fl_ps_mode']:
            return globals()['FlPsProgramBuilder'](pass_ctx)
        elif attrs['ps_mode'] == DistributedMode.SYNC:
            return globals()['CpuSyncPsProgramBuilder'](pass_ctx)
        else:
            return globals()['CpuAsyncPsProgramBuilder'](pass_ctx)