import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Optional
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.repo_config import FeastConfigError, load_repo_config

def _test_config(config_text, expect_error: Optional[str]):
    if False:
        for i in range(10):
            print('nop')
    '\n    Try loading a repo config and check raised error against a regex.\n    '
    with tempfile.TemporaryDirectory() as repo_dir_name:
        repo_path = Path(repo_dir_name)
        repo_config = repo_path / 'feature_store.yaml'
        repo_config.write_text(config_text)
        error = None
        rc = None
        try:
            rc = load_repo_config(repo_path, repo_config)
        except FeastConfigError as e:
            error = e
        if expect_error is not None:
            assert expect_error in str(error)
        else:
            print(f'error: {error}')
            assert error is None
        return rc

def test_nullable_online_store_aws():
    if False:
        print('Hello World!')
    _test_config(dedent('\n        project: foo\n        registry: "registry.db"\n        provider: aws\n        online_store: null\n        entity_key_serialization_version: 2\n        '), expect_error='__root__ -> offline_store -> __root__\n  please specify either cluster_id & user if using provisioned clusters, or workgroup if using serverless (type=value_error)')

def test_nullable_online_store_gcp():
    if False:
        for i in range(10):
            print('nop')
    _test_config(dedent('\n        project: foo\n        registry: "registry.db"\n        provider: gcp\n        online_store: null\n        entity_key_serialization_version: 2\n        '), expect_error=None)

def test_nullable_online_store_local():
    if False:
        i = 10
        return i + 15
    _test_config(dedent('\n        project: foo\n        registry: "registry.db"\n        provider: local\n        online_store: null\n        entity_key_serialization_version: 2\n        '), expect_error=None)

def test_local_config():
    if False:
        while True:
            i = 10
    _test_config(dedent('\n        project: foo\n        registry: "registry.db"\n        provider: local\n        entity_key_serialization_version: 2\n        '), expect_error=None)

def test_local_config_with_full_online_class():
    if False:
        return 10
    c = _test_config(dedent('\n        project: foo\n        registry: "registry.db"\n        provider: local\n        online_store:\n            type: feast.infra.online_stores.sqlite.SqliteOnlineStore\n        entity_key_serialization_version: 2\n        '), expect_error=None)
    assert isinstance(c.online_store, SqliteOnlineStoreConfig)

def test_local_config_with_full_online_class_directly():
    if False:
        i = 10
        return i + 15
    c = _test_config(dedent('\n        project: foo\n        registry: "registry.db"\n        provider: local\n        online_store: feast.infra.online_stores.sqlite.SqliteOnlineStore\n        entity_key_serialization_version: 2\n        '), expect_error=None)
    assert isinstance(c.online_store, SqliteOnlineStoreConfig)

def test_gcp_config():
    if False:
        return 10
    _test_config(dedent('\n        project: foo\n        registry: gs://registry.db\n        provider: gcp\n        entity_key_serialization_version: 2\n        '), expect_error=None)

def test_extra_field():
    if False:
        return 10
    _test_config(dedent('\n        project: foo\n        registry: "registry.db"\n        provider: local\n        online_store:\n            type: sqlite\n            that_field_should_not_be_here: yes\n            path: "online_store.db"\n        '), expect_error='__root__ -> online_store -> that_field_should_not_be_here\n  extra fields not permitted (type=value_error.extra)')

def test_no_online_store_type():
    if False:
        i = 10
        return i + 15
    _test_config(dedent('\n        project: foo\n        registry: "registry.db"\n        provider: local\n        online_store:\n            path: "blah"\n        entity_key_serialization_version: 2\n        '), expect_error=None)

def test_bad_type():
    if False:
        print('Hello World!')
    _test_config(dedent('\n        project: foo\n        registry: "registry.db"\n        provider: local\n        online_store:\n            path: 100500\n        '), expect_error='__root__ -> online_store -> path\n  str type expected')

def test_no_project():
    if False:
        return 10
    _test_config(dedent('\n        registry: "registry.db"\n        provider: local\n        online_store:\n            path: foo\n        entity_key_serialization_version: 2\n        '), expect_error='1 validation error for RepoConfig\nproject\n  field required (type=value_error.missing)')

def test_invalid_project_name():
    if False:
        return 10
    _test_config(dedent('\n        project: foo-1\n        registry: "registry.db"\n        provider: local\n        '), expect_error='alphanumerical values ')
    _test_config(dedent('\n        project: _foo\n        registry: "registry.db"\n        provider: local\n        '), expect_error='alphanumerical values ')