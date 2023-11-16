from dagster import repository
from dagster._core.definitions.cacheable_assets import CacheableAssetsDefinition

class FooCacheableAssetsDefinition(CacheableAssetsDefinition):

    def compute_cacheable_data(self):
        if False:
            while True:
                i = 10
        return []

    def build_definitions(self, *_args, **_kwargs):
        if False:
            for i in range(10):
                print('nop')
        return []

@repository
def single_pending_repository():
    if False:
        return 10
    return [FooCacheableAssetsDefinition('abc')]