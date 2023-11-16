from __future__ import annotations
import semver

def object_exists(path: str):
    if False:
        print('Hello World!')
    'Returns true if importable python object is there.'
    from airflow.utils.module_loading import import_string
    try:
        import_string(path)
        return True
    except ImportError:
        return False

def get_provider_version(provider_name):
    if False:
        while True:
            i = 10
    '\n    Returns provider version given provider package name.\n\n    Example::\n        if provider_version(\'apache-airflow-providers-cncf-kubernetes\') >= (6, 0):\n            raise Exception(\n                "You must now remove `get_kube_client` from PodManager "\n                "and make kube_client a required argument."\n            )\n    '
    from airflow.providers_manager import ProvidersManager
    info = ProvidersManager().providers[provider_name]
    return semver.VersionInfo.parse(info.version)

def get_provider_min_airflow_version(provider_name):
    if False:
        return 10
    from airflow.providers_manager import ProvidersManager
    p = ProvidersManager()
    deps = p.providers[provider_name].data['dependencies']
    airflow_dep = next((x for x in deps if x.startswith('apache-airflow')))
    min_airflow_version = tuple(map(int, airflow_dep.split('>=')[1].split('.')))
    return min_airflow_version