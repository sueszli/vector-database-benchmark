from dagster import asset
from dagster._core.definitions.asset_graph import AssetGraph

@asset
def iris_dataset():
    if False:
        print('Hello World!')
    return 1
from dagstermill import define_dagstermill_asset
from dagster import AssetIn, Field, Int, file_relative_path
iris_kmeans_jupyter_notebook = define_dagstermill_asset(name='iris_kmeans_jupyter', notebook_path=file_relative_path(__file__, './notebooks/iris-kmeans.ipynb'), group_name='template_tutorial', ins={'iris': AssetIn('iris_dataset')}, config_schema=Field(Int, default_value=3, is_required=False, description='The number of clusters to find'))
from dagstermill import ConfigurableLocalOutputNotebookIOManager
from dagster import AssetSelection, define_asset_job, with_resources
assets_with_resource = with_resources([iris_kmeans_jupyter_notebook, iris_dataset], resource_defs={'output_notebook_io_manager': ConfigurableLocalOutputNotebookIOManager()})
config_asset_job = define_asset_job(name='config_asset_job', selection=AssetSelection.assets(iris_kmeans_jupyter_notebook).upstream()).resolve(asset_graph=AssetGraph.from_assets(assets_with_resource))