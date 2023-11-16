import inspect
from collections import OrderedDict
from typing import List
from ray import cloudpickle
from ray.dag import PARENT_CLASS_NODE_KEY, ClassMethodNode, ClassNode, DAGNode
from ray.dag.function_node import FunctionNode
from ray.dag.input_node import InputNode
from ray.dag.utils import _DAGNodeNameGenerator
from ray.experimental.gradio_utils import type_to_string
from ray.serve._private.constants import RAY_SERVE_ENABLE_NEW_HANDLE_API, SERVE_DEFAULT_APP_NAME
from ray.serve._private.deployment_executor_node import DeploymentExecutorNode
from ray.serve._private.deployment_function_executor_node import DeploymentFunctionExecutorNode
from ray.serve._private.deployment_function_node import DeploymentFunctionNode
from ray.serve._private.deployment_method_executor_node import DeploymentMethodExecutorNode
from ray.serve._private.deployment_method_node import DeploymentMethodNode
from ray.serve._private.deployment_node import DeploymentNode
from ray.serve.deployment import Deployment, schema_to_deployment
from ray.serve.deployment_graph import RayServeDAGHandle
from ray.serve.handle import DeploymentHandle, RayServeHandle
from ray.serve.schema import DeploymentSchema

def build(ray_dag_root_node: DAGNode, name: str=SERVE_DEFAULT_APP_NAME) -> List[Deployment]:
    if False:
        while True:
            i = 10
    "Do all the DAG transformation, extraction and generation needed to\n    produce a runnable and deployable serve pipeline application from a valid\n    DAG authored with Ray DAG API.\n\n    This should be the only user facing API that user interacts with.\n\n    Assumptions:\n        Following enforcements are only applied at generating and applying\n        pipeline artifact, but not blockers for local development and testing.\n\n        - ALL args and kwargs used in DAG building should be JSON serializable.\n            This means in order to ensure your pipeline application can run on\n            a remote cluster potentially with different runtime environment,\n            among all options listed:\n\n                1) binding in-memory objects\n                2) Rely on pickling\n                3) Enforce JSON serialibility on all args used\n\n            We believe both 1) & 2) rely on unstable in-memory objects or\n            cross version pickling / closure capture, where JSON serialization\n            provides the right contract needed for proper deployment.\n\n        - ALL classes and methods used should be visible on top of the file and\n            importable via a fully qualified name. Thus no inline class or\n            function definitions should be used.\n\n    Args:\n        ray_dag_root_node: DAGNode acting as root of a Ray authored DAG. It\n            should be executable via `ray_dag_root_node.execute(user_input)`\n            and should have `InputNode` in it.\n        name: Application name,. If provided, formatting all the deployment name to\n            {name}_{deployment_name}, if not provided, the deployment name won't be\n            updated.\n\n    Returns:\n        deployments: All deployments needed for an e2e runnable serve pipeline,\n            accessible via python .remote() call.\n\n    Examples:\n\n        .. code-block:: python\n\n            with InputNode() as dag_input:\n                m1 = Model.bind(1)\n                m2 = Model.bind(2)\n                m1_output = m1.forward.bind(dag_input[0])\n                m2_output = m2.forward.bind(dag_input[1])\n                ray_dag = ensemble.bind(m1_output, m2_output)\n\n        Assuming we have non-JSON serializable or inline defined class or\n        function in local pipeline development.\n\n        .. code-block:: python\n\n            from ray.serve.api import build as build_app\n            deployments = build_app(ray_dag) # it can be method node\n            deployments = build_app(m1) # or just a regular node.\n    "
    with _DAGNodeNameGenerator() as node_name_generator:
        serve_root_dag = ray_dag_root_node.apply_recursive(lambda node: transform_ray_dag_to_serve_dag(node, node_name_generator, name))
    deployments = extract_deployments_from_serve_dag(serve_root_dag)
    if isinstance(serve_root_dag, DeploymentFunctionNode) and len(deployments) != 1:
        raise ValueError("The ingress deployment to your application cannot be a function if there are multiple deployments. If you want to compose them, use a class. If you're using the DAG API, the function should be bound to a DAGDriver.")
    serve_executor_root_dag = serve_root_dag.apply_recursive(transform_serve_dag_to_serve_executor_dag)
    root_driver_deployment = deployments[-1]
    new_driver_deployment = generate_executor_dag_driver_deployment(serve_executor_root_dag, root_driver_deployment)
    deployments[-1] = new_driver_deployment
    deployments_with_http = process_ingress_deployment_in_serve_dag(deployments)
    return deployments_with_http

def get_and_validate_ingress_deployment(deployments: List[Deployment]) -> Deployment:
    if False:
        print('Hello World!')
    'Validation for http route prefixes for a list of deployments in pipeline.\n\n    Ensures:\n        1) One and only one ingress deployment with given route prefix.\n        2) All other not ingress deployments should have prefix of None.\n    '
    ingress_deployments = []
    for deployment in deployments:
        if deployment.route_prefix is not None:
            ingress_deployments.append(deployment)
    if len(ingress_deployments) != 1:
        raise ValueError(f'Only one deployment in an Serve Application or DAG can have non-None route prefix. {len(ingress_deployments)} ingress deployments found: {ingress_deployments}')
    return ingress_deployments[0]

def transform_ray_dag_to_serve_dag(dag_node: DAGNode, node_name_generator: _DAGNodeNameGenerator, app_name: str):
    if False:
        print('Hello World!')
    '\n    Transform a Ray DAG to a Serve DAG. Map ClassNode to DeploymentNode with\n    ray decorated body passed in, and ClassMethodNode to DeploymentMethodNode.\n    When provided name, all Deployment name will {name}_{deployment_name}\n    '
    if isinstance(dag_node, ClassNode):
        deployment_name = node_name_generator.get_node_name(dag_node)

        def replace_with_handle(node):
            if False:
                while True:
                    i = 10
            if isinstance(node, DeploymentNode) or isinstance(node, DeploymentFunctionNode):
                if RAY_SERVE_ENABLE_NEW_HANDLE_API:
                    return DeploymentHandle(node._deployment.name, app_name, sync=False)
                else:
                    return RayServeHandle(node._deployment.name, app_name, sync=False)
            elif isinstance(node, DeploymentExecutorNode):
                return node._deployment_handle
        (replaced_deployment_init_args, replaced_deployment_init_kwargs) = dag_node.apply_functional([dag_node.get_args(), dag_node.get_kwargs()], predictate_fn=lambda node: isinstance(node, (DeploymentNode, DeploymentMethodNode, DeploymentFunctionNode, DeploymentExecutorNode, DeploymentFunctionExecutorNode, DeploymentMethodExecutorNode)), apply_fn=replace_with_handle)
        deployment_schema: DeploymentSchema = dag_node._bound_other_args_to_resolve['deployment_schema']
        deployment_shell: Deployment = schema_to_deployment(deployment_schema)
        if inspect.isclass(dag_node._body) and deployment_shell.name != dag_node._body.__name__:
            deployment_name = deployment_shell.name
        if deployment_shell.route_prefix is None or deployment_shell.route_prefix != f'/{deployment_shell.name}':
            route_prefix = deployment_shell.route_prefix
        else:
            route_prefix = f'/{deployment_name}'
        deployment = deployment_shell.options(func_or_class=dag_node._body, name=deployment_name, route_prefix=route_prefix, _init_args=replaced_deployment_init_args, _init_kwargs=replaced_deployment_init_kwargs, _internal=True)
        return DeploymentNode(deployment, app_name, dag_node.get_args(), dag_node.get_kwargs(), dag_node.get_options(), other_args_to_resolve=dag_node.get_other_args_to_resolve())
    elif isinstance(dag_node, ClassMethodNode):
        other_args_to_resolve = dag_node.get_other_args_to_resolve()
        parent_deployment_node = other_args_to_resolve[PARENT_CLASS_NODE_KEY]
        parent_class = parent_deployment_node._deployment.func_or_class
        method = getattr(parent_class, dag_node._method_name)
        if 'return' in method.__annotations__:
            other_args_to_resolve['result_type_string'] = type_to_string(method.__annotations__['return'])
        return DeploymentMethodNode(parent_deployment_node._deployment, dag_node._method_name, app_name, dag_node.get_args(), dag_node.get_kwargs(), dag_node.get_options(), other_args_to_resolve=other_args_to_resolve)
    elif isinstance(dag_node, FunctionNode) and dag_node.get_other_args_to_resolve().get('is_from_serve_deployment'):
        deployment_name = node_name_generator.get_node_name(dag_node)
        other_args_to_resolve = dag_node.get_other_args_to_resolve()
        if 'return' in dag_node._body.__annotations__:
            other_args_to_resolve['result_type_string'] = type_to_string(dag_node._body.__annotations__['return'])
        if 'deployment_schema' in dag_node._bound_other_args_to_resolve:
            schema = dag_node._bound_other_args_to_resolve['deployment_schema']
            if inspect.isfunction(dag_node._body) and schema.name != dag_node._body.__name__:
                deployment_name = schema.name
        return DeploymentFunctionNode(dag_node._body, deployment_name, app_name, dag_node.get_args(), dag_node.get_kwargs(), dag_node.get_options(), other_args_to_resolve=other_args_to_resolve)
    else:
        return dag_node

def extract_deployments_from_serve_dag(serve_dag_root: DAGNode) -> List[Deployment]:
    if False:
        return 10
    'Extract deployment python objects from a transformed serve DAG. Should\n    only be called after `transform_ray_dag_to_serve_dag`, otherwise nothing\n    to return.\n\n    Args:\n        serve_dag_root: Transformed serve dag root node.\n    Returns:\n        deployments (List[Deployment]): List of deployment python objects\n            fetched from serve dag.\n    '
    deployments = OrderedDict()

    def extractor(dag_node):
        if False:
            while True:
                i = 10
        if isinstance(dag_node, (DeploymentNode, DeploymentFunctionNode)):
            deployment = dag_node._deployment
            deployments[deployment.name] = deployment
        return dag_node
    serve_dag_root.apply_recursive(extractor)
    return list(deployments.values())

def transform_serve_dag_to_serve_executor_dag(serve_dag_root_node: DAGNode):
    if False:
        i = 10
        return i + 15
    'Given a runnable serve dag with deployment init args and options\n    processed, transform into an equivalent, but minimal dag optimized for\n    execution.\n    '
    if isinstance(serve_dag_root_node, DeploymentNode):
        return DeploymentExecutorNode(serve_dag_root_node._deployment_handle, serve_dag_root_node.get_args(), serve_dag_root_node.get_kwargs())
    elif isinstance(serve_dag_root_node, DeploymentMethodNode):
        return DeploymentMethodExecutorNode(serve_dag_root_node._deployment_method_name, serve_dag_root_node.get_args(), serve_dag_root_node.get_kwargs(), other_args_to_resolve=serve_dag_root_node.get_other_args_to_resolve())
    elif isinstance(serve_dag_root_node, DeploymentFunctionNode):
        return DeploymentFunctionExecutorNode(serve_dag_root_node._deployment_handle, serve_dag_root_node.get_args(), serve_dag_root_node.get_kwargs(), other_args_to_resolve=serve_dag_root_node.get_other_args_to_resolve())
    else:
        return serve_dag_root_node

def generate_executor_dag_driver_deployment(serve_executor_dag_root_node: DAGNode, original_driver_deployment: Deployment):
    if False:
        i = 10
        return i + 15
    "Given a transformed minimal execution serve dag, and original DAGDriver\n    deployment, generate new DAGDriver deployment that uses new serve executor\n    dag as init_args.\n\n    Args:\n        serve_executor_dag_root_node: Transformed\n            executor serve dag with only barebone deployment handles.\n        original_driver_deployment: User's original DAGDriver\n            deployment that wrapped Ray DAG as init args.\n    Returns:\n        executor_dag_driver_deployment: New DAGDriver deployment\n            with executor serve dag as init args.\n    "

    def replace_with_handle(node):
        if False:
            return 10
        if isinstance(node, DeploymentExecutorNode):
            return node._deployment_handle
        elif isinstance(node, DeploymentFunctionExecutorNode):
            if len(node.get_args()) == 0 and len(node.get_kwargs()) == 0:
                return node._deployment_function_handle
            else:
                return RayServeDAGHandle(cloudpickle.dumps(node))
        elif isinstance(node, DeploymentMethodExecutorNode):
            return RayServeDAGHandle(cloudpickle.dumps(node))
    (replaced_deployment_init_args, replaced_deployment_init_kwargs) = serve_executor_dag_root_node.apply_functional([serve_executor_dag_root_node.get_args(), serve_executor_dag_root_node.get_kwargs()], predictate_fn=lambda node: isinstance(node, (DeploymentExecutorNode, DeploymentFunctionExecutorNode, DeploymentMethodExecutorNode)), apply_fn=replace_with_handle)
    return original_driver_deployment.options(_init_args=replaced_deployment_init_args, _init_kwargs=replaced_deployment_init_kwargs, _internal=True)

def get_pipeline_input_node(serve_dag_root_node: DAGNode):
    if False:
        print('Hello World!')
    "Return the InputNode singleton node from serve dag, and throw\n    exceptions if we didn't find any, or found more than one.\n\n    Args:\n        ray_dag_root_node: DAGNode acting as root of a Ray authored DAG. It\n            should be executable via `ray_dag_root_node.execute(user_input)`\n            and should have `InputNode` in it.\n    Returns\n        pipeline_input_node: Singleton input node for the serve pipeline.\n    "
    input_nodes = []

    def extractor(dag_node):
        if False:
            print('Hello World!')
        if isinstance(dag_node, InputNode):
            input_nodes.append(dag_node)
    serve_dag_root_node.apply_recursive(extractor)
    assert len(input_nodes) == 1, f'There should be one and only one InputNode in the DAG. Found {len(input_nodes)} InputNode(s) instead.'
    return input_nodes[0]

def process_ingress_deployment_in_serve_dag(deployments: List[Deployment]) -> List[Deployment]:
    if False:
        i = 10
        return i + 15
    'Mark the last fetched deployment in a serve dag as exposed with default\n    prefix.\n    '
    if len(deployments) == 0:
        return deployments
    ingress_deployment = deployments[-1]
    if ingress_deployment.route_prefix in [None, f'/{ingress_deployment.name}']:
        new_ingress_deployment = ingress_deployment.options(route_prefix='/', _internal=True)
        deployments[-1] = new_ingress_deployment
    for (i, deployment) in enumerate(deployments[:-1]):
        if deployment.route_prefix is not None and deployment.route_prefix != f'/{deployment.name}':
            raise ValueError(f'Route prefix is only configurable on the ingress deployment. Please do not set non-default route prefix: {deployment.route_prefix} on non-ingress deployment of the serve DAG. ')
        else:
            deployments[i] = deployment.options(route_prefix=None, _internal=True)
    return deployments