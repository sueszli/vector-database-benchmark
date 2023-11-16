from __future__ import annotations
from typing import TYPE_CHECKING
import pluggy
local_settings_hookspec = pluggy.HookspecMarker('airflow.policy')
hookimpl = pluggy.HookimplMarker('airflow.policy')
__all__: list[str] = ['hookimpl']
if TYPE_CHECKING:
    from airflow.models.baseoperator import BaseOperator
    from airflow.models.dag import DAG
    from airflow.models.taskinstance import TaskInstance

@local_settings_hookspec
def task_policy(task: BaseOperator) -> None:
    if False:
        while True:
            i = 10
    "\n    Allow altering tasks after they are loaded in the DagBag.\n\n    It allows administrator to rewire some task's parameters.  Alternatively you can raise\n    ``AirflowClusterPolicyViolation`` exception to stop DAG from being executed.\n\n    Here are a few examples of how this can be useful:\n\n    * You could enforce a specific queue (say the ``spark`` queue) for tasks using the ``SparkOperator`` to\n      make sure that these tasks get wired to the right workers\n    * You could enforce a task timeout policy, making sure that no tasks run for more than 48 hours\n\n    :param task: task to be mutated\n    "

@local_settings_hookspec
def dag_policy(dag: DAG) -> None:
    if False:
        print('Hello World!')
    "\n    Allow altering DAGs after they are loaded in the DagBag.\n\n    It allows administrator to rewire some DAG's parameters.\n    Alternatively you can raise ``AirflowClusterPolicyViolation`` exception\n    to stop DAG from being executed.\n\n    Here are a few examples of how this can be useful:\n\n    * You could enforce default user for DAGs\n    * Check if every DAG has configured tags\n\n    :param dag: dag to be mutated\n    "

@local_settings_hookspec
def task_instance_mutation_hook(task_instance: TaskInstance) -> None:
    if False:
        print('Hello World!')
    '\n    Allow altering task instances before being queued by the Airflow scheduler.\n\n    This could be used, for instance, to modify the task instance during retries.\n\n    :param task_instance: task instance to be mutated\n    '

@local_settings_hookspec
def pod_mutation_hook(pod) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Mutate pod before scheduling.\n\n    This setting allows altering ``kubernetes.client.models.V1Pod`` object before they are passed to the\n    Kubernetes client for scheduling.\n\n    This could be used, for instance, to add sidecar or init containers to every worker pod launched by\n    KubernetesExecutor or KubernetesPodOperator.\n    '

@local_settings_hookspec(firstresult=True)
def get_airflow_context_vars(context) -> dict[str, str]:
    if False:
        i = 10
        return i + 15
    '\n    Inject airflow context vars into default airflow context vars.\n\n    This setting allows getting the airflow context vars, which are key value pairs.  They are then injected\n    to default airflow context vars, which in the end are available as environment variables when running\n    tasks dag_id, task_id, execution_date, dag_run_id, try_number are reserved keys.\n\n    :param context: The context for the task_instance of interest.\n    '

@local_settings_hookspec(firstresult=True)
def get_dagbag_import_timeout(dag_file_path: str) -> int | float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Allow for dynamic control of the DAG file parsing timeout based on the DAG file path.\n\n    It is useful when there are a few DAG files requiring longer parsing times, while others do not.\n    You can control them separately instead of having one value for all DAG files.\n\n    If the return value is less than or equal to 0, it means no timeout during the DAG parsing.\n    '

class DefaultPolicy:
    """Default implementations of the policy functions.

    :meta private:
    """

    @staticmethod
    @hookimpl
    def get_dagbag_import_timeout(dag_file_path: str):
        if False:
            print('Hello World!')
        from airflow.configuration import conf
        return conf.getfloat('core', 'DAGBAG_IMPORT_TIMEOUT')

    @staticmethod
    @hookimpl
    def get_airflow_context_vars(context):
        if False:
            print('Hello World!')
        return {}

def make_plugin_from_local_settings(pm: pluggy.PluginManager, module, names: set[str]):
    if False:
        while True:
            i = 10
    '\n    Turn the functions from airflow_local_settings module into a custom/local plugin.\n\n    Allows plugin-registered functions to co-operate with pluggy/setuptool\n    entrypoint plugins of the same methods.\n\n    Airflow local settings will be "win" (i.e. they have the final say) as they are the last plugin\n    registered.\n\n    :meta private:\n    '
    import inspect
    import textwrap
    import attr
    hook_methods = set()

    def _make_shim_fn(name, desired_sig, target):
        if False:
            print('Hello World!')
        codestr = textwrap.dedent(f"\n            def {name}_name_mismatch_shim{desired_sig}:\n                return __target({' ,'.join(desired_sig.parameters)})\n            ")
        code = compile(codestr, '<policy-shim>', 'single')
        scope = {'__target': target}
        exec(code, scope, scope)
        return scope[f'{name}_name_mismatch_shim']

    @attr.define(frozen=True)
    class AirflowLocalSettingsPolicy:
        hook_methods: tuple[str, ...]
        __name__ = 'AirflowLocalSettingsPolicy'

        def __dir__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.hook_methods
    for name in names:
        if not hasattr(pm.hook, name):
            continue
        policy = getattr(module, name)
        if not policy:
            continue
        local_sig = inspect.signature(policy)
        policy_sig = inspect.signature(globals()[name])
        if local_sig.parameters.keys() != policy_sig.parameters.keys():
            policy = _make_shim_fn(name, policy_sig, target=policy)
        setattr(AirflowLocalSettingsPolicy, name, staticmethod(hookimpl(policy, specname=name)))
        hook_methods.add(name)
    if hook_methods:
        pm.register(AirflowLocalSettingsPolicy(hook_methods=tuple(hook_methods)))
    return hook_methods