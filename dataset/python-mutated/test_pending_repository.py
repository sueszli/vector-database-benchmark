import dagster._check as check
import pytest
from dagster import AssetKey, AssetsDefinition, AssetSelection, DagsterInvalidDefinitionError, JobDefinition, RepositoryDefinition, asset, define_asset_job, op, repository, resource, with_resources
from dagster._core.definitions.cacheable_assets import AssetsDefinitionCacheableData, CacheableAssetsDefinition
from dagster._core.definitions.repository_definition import PendingRepositoryDefinition, RepositoryLoadData
from .test_repository import define_empty_job, define_simple_job, define_with_resources_job

def define_cacheable_and_uncacheable_assets():
    if False:
        return 10

    class MyCacheableAssets(CacheableAssetsDefinition):

        def compute_cacheable_data(self):
            if False:
                return 10
            return [AssetsDefinitionCacheableData(keys_by_input_name={'upstream': AssetKey('upstream')}, keys_by_output_name={'result': AssetKey(self.unique_id)})]

        def build_definitions(self, data):
            if False:
                return 10

            @op(name=self.unique_id)
            def _op(upstream):
                if False:
                    for i in range(10):
                        print('nop')
                return upstream + 1
            return [AssetsDefinition.from_op(_op, keys_by_input_name=cd.keys_by_input_name, keys_by_output_name=cd.keys_by_output_name) for cd in data]

    @asset
    def upstream():
        if False:
            for i in range(10):
                print('nop')
        return 1

    @asset
    def downstream(a, b):
        if False:
            for i in range(10):
                print('nop')
        return a + b
    return [MyCacheableAssets('a'), MyCacheableAssets('b'), upstream, downstream]

@repository
def pending_repo():
    if False:
        i = 10
        return i + 15
    return [define_empty_job(), define_simple_job(), *define_with_resources_job(), define_cacheable_and_uncacheable_assets(), define_asset_job('all_asset_job', selection=AssetSelection.keys(AssetKey('a'), AssetKey('b'), AssetKey('upstream'), AssetKey('downstream')))]

def test_resolve_empty():
    if False:
        print('Hello World!')
    assert isinstance(pending_repo, PendingRepositoryDefinition)
    with pytest.raises(check.CheckError):
        repo = pending_repo.reconstruct_repository_definition(repository_load_data=None)
    repo = pending_repo.compute_repository_definition()
    assert isinstance(repo, RepositoryDefinition)
    assert isinstance(repo.get_job('simple_job'), JobDefinition)
    assert isinstance(repo.get_job('all_asset_job'), JobDefinition)

def test_resolve_missing_key():
    if False:
        while True:
            i = 10
    assert isinstance(pending_repo, PendingRepositoryDefinition)
    with pytest.raises(check.CheckError, match='No metadata found'):
        pending_repo.reconstruct_repository_definition(repository_load_data=RepositoryLoadData(cached_data_by_key={'a': [AssetsDefinitionCacheableData(keys_by_input_name={'upstream': AssetKey('upstream')}, keys_by_output_name={'result': AssetKey('a')})]}))

def test_resolve_wrong_data():
    if False:
        while True:
            i = 10
    assert isinstance(pending_repo, PendingRepositoryDefinition)
    with pytest.raises(DagsterInvalidDefinitionError, match='Input asset .*\\"b\\".* is not produced by any of the provided asset ops and is not one of the provided sources'):
        pending_repo.reconstruct_repository_definition(repository_load_data=RepositoryLoadData(cached_data_by_key={'a': [AssetsDefinitionCacheableData(keys_by_input_name={'upstream': AssetKey('upstream')}, keys_by_output_name={'result': AssetKey('a')})], 'b': [AssetsDefinitionCacheableData(keys_by_input_name={'upstream': AssetKey('upstream')}, keys_by_output_name={'result': AssetKey('BAD_ASSET_KEY')})]}))

def define_resource_dependent_cacheable_and_uncacheable_assets():
    if False:
        return 10

    class ResourceDependentCacheableAsset(CacheableAssetsDefinition):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__('res_downstream')

        def compute_cacheable_data(self):
            if False:
                return 10
            return [AssetsDefinitionCacheableData(keys_by_input_name={'res_upstream': AssetKey('res_upstream')}, keys_by_output_name={'result': AssetKey('res_midstream')})]

        def build_definitions(self, data):
            if False:
                for i in range(10):
                    print('nop')

            @op(name='res_midstream', required_resource_keys={'foo'})
            def _op(context, res_upstream):
                if False:
                    i = 10
                    return i + 15
                return res_upstream + context.resources.foo
            return [AssetsDefinition.from_op(_op, keys_by_input_name=cd.keys_by_input_name, keys_by_output_name=cd.keys_by_output_name) for cd in data]

    @asset(required_resource_keys={'foo'})
    def res_upstream(context):
        if False:
            print('Hello World!')
        return context.resources.foo

    @asset(required_resource_keys={'foo'})
    def res_downstream(context, res_midstream):
        if False:
            return 10
        return res_midstream + context.resources.foo
    return [ResourceDependentCacheableAsset(), res_upstream, res_downstream]

def test_resolve_no_resources():
    if False:
        print('Hello World!')
    'Test that loading a repo with a resource-dependent cacheable asset fails if the resource is not\n    provided.\n    '
    with pytest.raises(DagsterInvalidDefinitionError):
        try:

            @repository
            def resource_dependent_repo_no_resources():
                if False:
                    i = 10
                    return i + 15
                return [define_resource_dependent_cacheable_and_uncacheable_assets(), define_asset_job('all_asset_job')]
            resource_dependent_repo_no_resources.compute_repository_definition()
        except DagsterInvalidDefinitionError as e:
            assert 'res_midstream' in str(e)
            raise e

def test_resolve_with_resources():
    if False:
        for i in range(10):
            print('nop')
    'Test that loading a repo with a resource-dependent cacheable asset succeeds if the resource is\n    provided.\n    '

    @resource
    def foo_resource():
        if False:
            i = 10
            return i + 15
        return 3

    @repository
    def resource_dependent_repo_with_resources():
        if False:
            while True:
                i = 10
        return [with_resources(define_resource_dependent_cacheable_and_uncacheable_assets(), {'foo': foo_resource}), define_asset_job('all_asset_job')]
    repo = resource_dependent_repo_with_resources.compute_repository_definition()
    assert isinstance(repo, RepositoryDefinition)
    assert isinstance(repo.get_job('all_asset_job'), JobDefinition)

def test_group_cached_assets():
    if False:
        return 10
    'Test that with_attributes works properly on cacheable assets.'

    class MyCacheableAssets(CacheableAssetsDefinition):

        def compute_cacheable_data(self):
            if False:
                for i in range(10):
                    print('nop')
            return [AssetsDefinitionCacheableData(keys_by_input_name={}, keys_by_output_name={'result': AssetKey(self.unique_id)})]

        def build_definitions(self, data):
            if False:
                print('Hello World!')

            @op(name=self.unique_id)
            def _op():
                if False:
                    i = 10
                    return i + 15
                return 5
            return [AssetsDefinition.from_op(_op, keys_by_input_name=cd.keys_by_input_name, keys_by_output_name=cd.keys_by_output_name) for cd in data]
    my_cacheable_assets_cool = MyCacheableAssets('foo').with_attributes(group_names_by_key={AssetKey('foo'): 'my_cool_group'})
    my_lame_group_sel = AssetSelection.groups('my_lame_group')
    assert len(my_lame_group_sel.resolve(my_cacheable_assets_cool.build_definitions(my_cacheable_assets_cool.compute_cacheable_data()))) == 0
    my_cool_group_sel = AssetSelection.groups('my_cool_group')
    assert len(my_cool_group_sel.resolve(my_cacheable_assets_cool.build_definitions(my_cacheable_assets_cool.compute_cacheable_data()))) == 1

def test_multiple_wrapped_cached_assets():
    if False:
        while True:
            i = 10
    'Test that multiple wrappers (with_attributes, with_resources) work properly on cacheable assets.'

    @resource
    def foo_resource():
        if False:
            while True:
                i = 10
        return 3
    my_cacheable_assets_with_group_and_asset = [x.with_attributes(output_asset_key_replacements={AssetKey('res_downstream'): AssetKey('res_downstream_too')}) for x in with_resources([x.with_attributes(group_names_by_key={AssetKey('res_midstream'): 'my_cool_group'}) for x in define_resource_dependent_cacheable_and_uncacheable_assets()], {'foo': foo_resource})]

    @repository
    def resource_dependent_repo_with_resources():
        if False:
            i = 10
            return i + 15
        return [my_cacheable_assets_with_group_and_asset, define_asset_job('all_asset_job')]
    repo = resource_dependent_repo_with_resources.compute_repository_definition()
    assert isinstance(repo, RepositoryDefinition)
    assert isinstance(repo.get_job('all_asset_job'), JobDefinition)
    my_cool_group_sel = AssetSelection.groups('my_cool_group')
    assert len(my_cool_group_sel.resolve(my_cacheable_assets_with_group_and_asset[0].build_definitions(my_cacheable_assets_with_group_and_asset[0].compute_cacheable_data()) + my_cacheable_assets_with_group_and_asset[1:])) == 1