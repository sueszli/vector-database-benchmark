import os
from .start_hook_base import RayOnSparkStartHook
from .utils import get_spark_session
import logging
import threading
import time
_logger = logging.getLogger(__name__)

def get_db_entry_point():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return databricks entry_point instance, it is for calling some\n    internal API in databricks runtime\n    '
    from dbruntime import UserNamespaceInitializer
    user_namespace_initializer = UserNamespaceInitializer.getOrCreate()
    return user_namespace_initializer.get_spark_entry_point()

def display_databricks_driver_proxy_url(spark_context, port, title):
    if False:
        print('Hello World!')
    '\n    This helper function create a proxy URL for databricks driver webapp forwarding.\n    In databricks runtime, user does not have permission to directly access web\n    service binding on driver machine port, but user can visit it by a proxy URL with\n    following format: "/driver-proxy/o/{orgId}/{clusterId}/{port}/".\n    '
    from dbruntime.display import displayHTML
    driverLocal = spark_context._jvm.com.databricks.backend.daemon.driver.DriverLocal
    commandContextTags = driverLocal.commandContext().get().toStringMap().apply('tags')
    orgId = commandContextTags.apply('orgId')
    clusterId = commandContextTags.apply('clusterId')
    proxy_link = f'/driver-proxy/o/{orgId}/{clusterId}/{port}/'
    proxy_url = f'https://dbc-dp-{orgId}.cloud.databricks.com{proxy_link}'
    print('To monitor and debug Ray from Databricks, view the dashboard at ')
    print(f' {proxy_url}')
    displayHTML(f'\n      <div style="margin-bottom: 16px">\n          <a href="{proxy_link}">\n              Open {title} in a new tab\n          </a>\n      </div>\n    ')
DATABRICKS_AUTO_SHUTDOWN_POLL_INTERVAL_SECONDS = 3
DATABRICKS_RAY_ON_SPARK_AUTOSHUTDOWN_MINUTES = 'DATABRICKS_RAY_ON_SPARK_AUTOSHUTDOWN_MINUTES'
_DATABRICKS_DEFAULT_TMP_DIR = '/local_disk0/tmp'

class DefaultDatabricksRayOnSparkStartHook(RayOnSparkStartHook):

    def get_default_temp_dir(self):
        if False:
            while True:
                i = 10
        return _DATABRICKS_DEFAULT_TMP_DIR

    def on_ray_dashboard_created(self, port):
        if False:
            print('Hello World!')
        display_databricks_driver_proxy_url(get_spark_session().sparkContext, port, 'Ray Cluster Dashboard')

    def on_cluster_created(self, ray_cluster_handler):
        if False:
            for i in range(10):
                print('nop')
        db_api_entry = get_db_entry_point()
        if ray_cluster_handler.autoscale:
            auto_shutdown_minutes = 0
        else:
            auto_shutdown_minutes = float(os.environ.get(DATABRICKS_RAY_ON_SPARK_AUTOSHUTDOWN_MINUTES, '30'))
        if auto_shutdown_minutes == 0:
            _logger.info('The Ray cluster will keep running until you manually detach the Databricks notebook or call `ray.util.spark.shutdown_ray_cluster()`.')
            return
        if auto_shutdown_minutes < 0:
            raise ValueError(f"You must set '{DATABRICKS_RAY_ON_SPARK_AUTOSHUTDOWN_MINUTES}' to a value >= 0.")
        try:
            db_api_entry.getIdleTimeMillisSinceLastNotebookExecution()
        except Exception:
            _logger.warning('Failed to retrieve idle time since last notebook execution, so that we cannot automatically shut down Ray cluster when Databricks notebook is inactive for the specified minutes. You need to manually detach Databricks notebook or call `ray.util.spark.shutdown_ray_cluster()` to shut down Ray cluster on spark.')
            return
        _logger.info(f"The Ray cluster will be shut down automatically if you don't run commands on the Databricks notebook for {auto_shutdown_minutes} minutes. You can change the auto-shutdown minutes by setting '{DATABRICKS_RAY_ON_SPARK_AUTOSHUTDOWN_MINUTES}' environment variable, setting it to 0 means that the Ray cluster keeps running until you manually call `ray.util.spark.shutdown_ray_cluster()` or detach Databricks notebook.")

        def auto_shutdown_watcher():
            if False:
                for i in range(10):
                    print('nop')
            auto_shutdown_millis = auto_shutdown_minutes * 60 * 1000
            while True:
                if ray_cluster_handler.is_shutdown:
                    return
                idle_time = db_api_entry.getIdleTimeMillisSinceLastNotebookExecution()
                if idle_time > auto_shutdown_millis:
                    from ray.util.spark import cluster_init
                    with cluster_init._active_ray_cluster_rwlock:
                        if ray_cluster_handler is cluster_init._active_ray_cluster:
                            cluster_init.shutdown_ray_cluster()
                    return
                time.sleep(DATABRICKS_AUTO_SHUTDOWN_POLL_INTERVAL_SECONDS)
        threading.Thread(target=auto_shutdown_watcher, daemon=True).start()