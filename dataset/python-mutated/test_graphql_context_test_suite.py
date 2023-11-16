import inspect
import sys
import dagster._check as check
import pytest
from dagster._core.test_utils import ExplodingRunLauncher
from .graphql_context_test_suite import GraphQLContextVariant, manage_graphql_context

@pytest.mark.graphql_context_variants
@pytest.mark.parametrize('variant', GraphQLContextVariant.all_non_launchable_variants())
def test_non_launchable_variants(variant):
    if False:
        while True:
            i = 10
    assert isinstance(variant, GraphQLContextVariant)
    with manage_graphql_context(variant) as context:
        assert isinstance(context.instance.run_launcher, ExplodingRunLauncher)

def get_all_static_functions(klass):
    if False:
        while True:
            i = 10
    check.invariant(sys.version_info >= (3,))

    def _yield_all():
        if False:
            for i in range(10):
                print('nop')
        for attr_name in dir(klass):
            attr = inspect.getattr_static(klass, attr_name)
            if isinstance(attr, staticmethod):
                yield attr.__func__
    return list(_yield_all())

def test_get_all_static_members():
    if False:
        i = 10
        return i + 15

    class Bar:
        class_var = 'foo'

        @staticmethod
        def static_one():
            if False:
                return 10
            pass

        @staticmethod
        def static_two():
            if False:
                print('Hello World!')
            pass

        @classmethod
        def classthing(cls):
            if False:
                for i in range(10):
                    print('nop')
            pass
    assert set(get_all_static_functions(Bar)) == {Bar.static_one, Bar.static_two}

def test_all_variants_in_variants_function():
    if False:
        while True:
            i = 10
    'This grabs all pre-defined variants on GraphQLContextVariant (defined as static methods that\n    return a single ContextVariant) and tests two things:\n    1) They all contain a unique test_id\n    2) That the all_variants() static method returns *all* of them.\n    '
    variant_test_ids_declared_on_class = set()
    for static_function in get_all_static_functions(GraphQLContextVariant):
        maybe_variant = static_function()
        if isinstance(maybe_variant, GraphQLContextVariant):
            assert maybe_variant.test_id
            assert maybe_variant.test_id not in variant_test_ids_declared_on_class
            variant_test_ids_declared_on_class.add(maybe_variant.test_id)
    test_ids_returned_by_all_variants = {var.test_id for var in GraphQLContextVariant.all_variants()}
    assert test_ids_returned_by_all_variants == variant_test_ids_declared_on_class

def test_non_launchable_marks_filter():
    if False:
        return 10
    non_launchable_test_ids = {var.test_id for var in [GraphQLContextVariant.non_launchable_sqlite_instance_lazy_repository(), GraphQLContextVariant.non_launchable_sqlite_instance_multi_location(), GraphQLContextVariant.non_launchable_sqlite_instance_managed_grpc_env(), GraphQLContextVariant.non_launchable_sqlite_instance_deployed_grpc_env(), GraphQLContextVariant.non_launchable_postgres_instance_lazy_repository(), GraphQLContextVariant.non_launchable_postgres_instance_multi_location(), GraphQLContextVariant.non_launchable_postgres_instance_managed_grpc_env()]}
    assert {var.test_id for var in GraphQLContextVariant.all_non_launchable_variants()} == non_launchable_test_ids