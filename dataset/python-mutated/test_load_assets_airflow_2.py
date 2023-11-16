import os
import tempfile
import pytest
from airflow import __version__ as airflow_version
from airflow.models import DagBag
from dagster import AssetKey, asset, materialize
from dagster_airflow import load_assets_from_airflow_dag, make_ephemeral_airflow_db_resource
from dagster_airflow_tests.marks import requires_local_db
ASSET_DAG = '\nfrom airflow import models\n\nfrom airflow.operators.bash import BashOperator\n\nimport datetime\n\ndefault_args = {"start_date": datetime.datetime(2023, 2, 1)}\n\nwith models.DAG(\n    dag_id="asset_dag", default_args=default_args, schedule=\'0 0 * * *\', tags=[\'example\'],\n) as asset_dag:\n    foo = BashOperator(\n        task_id="foo", bash_command="echo foo"\n    )\n\n    bar = BashOperator(\n        task_id="bar", bash_command="echo bar"\n    )\n\n    biz = BashOperator(\n        task_id="biz", bash_command="echo biz"\n    )\n\n    baz = BashOperator(\n        task_id="baz", bash_command="echo baz"\n    )\n\nwith models.DAG(\n    dag_id="other_dag", default_args=default_args, schedule=\'0 0 * * *\', tags=[\'example\'],\n) as other_dag:\n    foo = BashOperator(\n        task_id="foo", bash_command="echo foo"\n    )\n'

@pytest.mark.skipif(airflow_version < '2.0.0', reason='requires airflow 2')
@requires_local_db
def test_load_assets_from_airflow_dag():
    if False:
        for i in range(10):
            print('nop')
    with tempfile.TemporaryDirectory(suffix='assets') as tmpdir_path:
        with open(os.path.join(tmpdir_path, 'dag.py'), 'wb') as f:
            f.write(bytes(ASSET_DAG.encode('utf-8')))
        dag_bag = DagBag(dag_folder=tmpdir_path)
        asset_dag = dag_bag.get_dag(dag_id='asset_dag')
        assets = load_assets_from_airflow_dag(dag=asset_dag, task_ids_by_asset_key={AssetKey('foo_asset'): {'foo', 'bar'}, AssetKey('biz_asset'): {'biz', 'baz'}}, upstream_dependencies_by_asset_key={AssetKey('foo_asset'): {AssetKey('new_upstream_asset')}})
        other_dag = dag_bag.get_dag(dag_id='other_dag')
        other_assets = load_assets_from_airflow_dag(dag=other_dag)
        resources = None
        if assets:
            first_asset = next(iter(assets))
            if 'io_manager' in first_asset.resource_defs:
                resources = {'io_manager': first_asset.resource_defs['io_manager']}

        @asset(resource_defs=resources)
        def new_upstream_asset():
            if False:
                print('Hello World!')
            return 1
        result = materialize([*assets, new_upstream_asset, *other_assets], partition_key='2023-02-01T00:00:00', resources={'airflow_db': make_ephemeral_airflow_db_resource()})
        assert result.success