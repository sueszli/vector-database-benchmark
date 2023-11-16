"""Python Level API to launch Docker Containers using Hagrid"""
from __future__ import annotations
from enum import Enum
import getpass
import inspect
import os
import subprocess
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union
import gevent
from .cli import str_to_bool
from .grammar import find_available_port
from .names import random_name
from .util import shell
try:
    from syft.abstract_node import NodeSideType
    from syft.abstract_node import NodeType
    from syft.protocol.data_protocol import stage_protocol_changes
    from syft.service.response import SyftError
except Exception:
    pass
DEFAULT_PORT = 8080
ClientAlias = Any

def read_stream(stream: subprocess.PIPE) -> None:
    if False:
        return 10
    while True:
        line = stream.readline()
        if not line:
            break
        print(line, end='')
        gevent.sleep(0)

def to_snake_case(name: str) -> str:
    if False:
        i = 10
        return i + 15
    return name.lower().replace(' ', '_')

def get_syft_client() -> Optional[Any]:
    if False:
        i = 10
        return i + 15
    try:
        import syft as sy
        return sy
    except Exception:
        pass
    return None

def container_exists(name: str) -> bool:
    if False:
        print('Hello World!')
    output = shell(f"docker ps -q -f name='{name}'")
    return len(output) > 0

def port_from_container(name: str, deployment_type: DeploymentType) -> Optional[int]:
    if False:
        return 10
    container_suffix = ''
    if deployment_type == DeploymentType.SINGLE_CONTAINER:
        container_suffix = '-worker-1'
    elif deployment_type == DeploymentType.CONTAINER_STACK:
        container_suffix = '-proxy-1'
    else:
        raise NotImplementedError(f'port_from_container not implemented for the deployment type:{deployment_type}')
    container_name = name + container_suffix
    output = shell(f'docker port {container_name}')
    if len(output) > 0:
        try:
            lines = output.split('\n')
            parts = lines[0].split(':')
            port = int(parts[1].strip())
            return port
        except Exception:
            return None
    return None

def container_exists_with(name: str, port: int) -> bool:
    if False:
        while True:
            i = 10
    output = shell(f'docker ps -q -f name={name} | xargs -n 1 docker port | grep 0.0.0.0:{port}')
    return len(output) > 0

def get_node_type(node_type: Optional[Union[str, NodeType]]) -> Optional[NodeType]:
    if False:
        return 10
    if node_type is None:
        node_type = os.environ.get('ORCHESTRA_NODE_TYPE', NodeType.DOMAIN)
    try:
        return NodeType(node_type)
    except ValueError:
        print(f'node_type: {node_type} is not a valid NodeType: {NodeType}')
    return None

def get_deployment_type(deployment_type: Optional[str]) -> Optional[DeploymentType]:
    if False:
        print('Hello World!')
    if deployment_type is None:
        deployment_type = os.environ.get('ORCHESTRA_DEPLOYMENT_TYPE', DeploymentType.PYTHON)
    if deployment_type == 'container':
        deployment_type = 'container_stack'
    try:
        return DeploymentType(deployment_type)
    except ValueError:
        print(f'deployment_type: {deployment_type} is not a valid DeploymentType: {DeploymentType}')
    return None

class DeploymentType(Enum):
    PYTHON = 'python'
    SINGLE_CONTAINER = 'single_container'
    CONTAINER_STACK = 'container_stack'
    K8S = 'k8s'
    PODMAN = 'podman'

class NodeHandle:

    def __init__(self, node_type: NodeType, deployment_type: DeploymentType, node_side_type: NodeSideType, name: str, port: Optional[int]=None, url: Optional[str]=None, python_node: Optional[Any]=None, shutdown: Optional[Callable]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.node_type = node_type
        self.name = name
        self.port = port
        self.url = url
        self.python_node = python_node
        self.shutdown = shutdown
        self.deployment_type = deployment_type
        self.node_side_type = node_side_type

    @property
    def client(self) -> Any:
        if False:
            print('Hello World!')
        if self.port:
            sy = get_syft_client()
            return sy.login_as_guest(url=self.url, port=self.port)
        elif self.deployment_type == DeploymentType.PYTHON:
            return self.python_node.get_guest_client(verbose=False)
        else:
            raise NotImplementedError(f'client not implemented for the deployment type:{self.deployment_type}')

    def login_as_guest(self, **kwargs: Any) -> ClientAlias:
        if False:
            return 10
        return self.client.login_as_guest(**kwargs)

    def login(self, email: Optional[str]=None, password: Optional[str]=None, **kwargs: Any) -> ClientAlias:
        if False:
            i = 10
            return i + 15
        if not email:
            email = input('Email: ')
        if not password:
            password = getpass.getpass('Password: ')
        return self.client.login(email=email, password=password, **kwargs)

    def register(self, name: str, email: Optional[str]=None, password: Optional[str]=None, password_verify: Optional[str]=None, institution: Optional[str]=None, website: Optional[str]=None) -> Any:
        if False:
            print('Hello World!')
        if not email:
            email = input('Email: ')
        if not password:
            password = getpass.getpass('Password: ')
        if not password_verify:
            password_verify = getpass.getpass('Confirm Password: ')
        if password != password_verify:
            return SyftError(message='Passwords do not match')
        client = self.client
        return client.register(name=name, email=email, password=password, institution=institution, password_verify=password_verify, website=website)

    def land(self) -> None:
        if False:
            return 10
        if self.deployment_type == DeploymentType.PYTHON:
            if self.shutdown:
                self.shutdown()
        else:
            Orchestra.land(self.name, deployment_type=self.deployment_type)

def deploy_to_python(node_type_enum: NodeType, deployment_type_enum: DeploymentType, port: Union[int, str], name: str, host: str, reset: bool, tail: bool, dev_mode: bool, processes: int, local_db: bool, node_side_type: NodeSideType, enable_warnings: bool) -> Optional[NodeHandle]:
    if False:
        for i in range(10):
            print('nop')
    sy = get_syft_client()
    if sy is None:
        return sy
    worker_classes = {NodeType.DOMAIN: sy.Domain, NodeType.NETWORK: sy.Gateway}
    if hasattr(sy, 'Enclave'):
        worker_classes[NodeType.ENCLAVE] = sy.Enclave
    if hasattr(NodeType, 'GATEWAY'):
        worker_classes[NodeType.GATEWAY] = sy.Gateway
    if dev_mode:
        print('Staging Protocol Changes...')
        stage_protocol_changes()
    if port:
        if port == 'auto':
            port = find_available_port(host='localhost', port=None, search=True)
        sig = inspect.signature(sy.serve_node)
        if 'node_type' in sig.parameters.keys():
            (start, stop) = sy.serve_node(name=name, host=host, port=port, reset=reset, dev_mode=dev_mode, tail=tail, node_type=node_type_enum, node_side_type=node_side_type, enable_warnings=enable_warnings)
        else:
            (start, stop) = sy.serve_node(name=name, host=host, port=port, reset=reset, dev_mode=dev_mode, tail=tail)
        start()
        return NodeHandle(node_type=node_type_enum, deployment_type=deployment_type_enum, name=name, port=port, url='http://localhost', shutdown=stop, node_side_type=node_side_type)
    else:
        if node_type_enum in worker_classes:
            worker_class = worker_classes[node_type_enum]
            sig = inspect.signature(worker_class.named)
            if 'node_type' in sig.parameters.keys():
                worker = worker_class.named(name=name, processes=processes, reset=reset, local_db=local_db, node_type=node_type_enum, node_side_type=node_side_type, enable_warnings=enable_warnings)
            else:
                worker = worker_class.named(name=name, processes=processes, reset=reset, local_db=local_db)
        else:
            raise NotImplementedError(f'node_type: {node_type_enum} is not supported')
        return NodeHandle(node_type=node_type_enum, deployment_type=deployment_type_enum, name=name, python_node=worker, node_side_type=node_side_type)

def deploy_to_k8s(node_type_enum: NodeType, deployment_type_enum: DeploymentType, name: str, node_side_type: NodeSideType) -> NodeHandle:
    if False:
        print('Hello World!')
    node_port = int(os.environ.get('NODE_PORT', f'{DEFAULT_PORT}'))
    return NodeHandle(node_type=node_type_enum, deployment_type=deployment_type_enum, name=name, port=node_port, url='http://localhost', node_side_type=node_side_type)

def deploy_to_podman(node_type_enum: NodeType, deployment_type_enum: DeploymentType, name: str, node_side_type: NodeSideType) -> NodeHandle:
    if False:
        for i in range(10):
            print('nop')
    node_port = int(os.environ.get('NODE_PORT', f'{DEFAULT_PORT}'))
    return NodeHandle(node_type=node_type_enum, deployment_type=deployment_type_enum, name=name, port=node_port, url='http://localhost', node_side_type=node_side_type)

def deploy_to_container(node_type_enum: NodeType, deployment_type_enum: DeploymentType, node_side_type: NodeSideType, reset: bool, cmd: bool, tail: bool, verbose: bool, tag: str, render: bool, dev_mode: bool, port: Union[int, str], name: str, enable_warnings: bool) -> Optional[NodeHandle]:
    if False:
        return 10
    if port == 'auto' or port is None:
        if container_exists(name=name):
            port = port_from_container(name=name, deployment_type=deployment_type_enum)
        else:
            port = find_available_port(host='localhost', port=DEFAULT_PORT, search=True)
    if reset:
        Orchestra.reset(name, deployment_type_enum)
    elif container_exists_with(name=name, port=port):
        return NodeHandle(node_type=node_type_enum, deployment_type=deployment_type_enum, name=name, port=port, url='http://localhost', node_side_type=node_side_type)
    commands = ['hagrid', 'launch']
    name = random_name() if not name else name
    commands.extend([name, node_type_enum.value])
    commands.append('to')
    commands.append(f'docker:{port}')
    if dev_mode:
        commands.append('--dev')
    if not enable_warnings:
        commands.append('--no-warnings')
    if deployment_type_enum == DeploymentType.SINGLE_CONTAINER:
        commands.append('--deployment-type=single_container')
    if cmd:
        commands.append('--cmd')
    if tail:
        commands.append('--tail')
    if verbose:
        commands.append('--verbose')
    if tag:
        commands.append(f'--tag={tag}')
    if render:
        commands.append('--render')
    USER = os.environ.get('USER', getpass.getuser())
    env = os.environ.copy()
    env['USER'] = USER
    process = subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    stdout_thread = gevent.spawn(read_stream, process.stdout)
    stderr_thread = gevent.spawn(read_stream, process.stderr)
    gevent.joinall([stdout_thread, stderr_thread], raise_error=True)
    if not cmd:
        return NodeHandle(node_type=node_type_enum, deployment_type=deployment_type_enum, name=name, port=port, url='http://localhost', node_side_type=node_side_type)
    return None

class Orchestra:

    @staticmethod
    def launch(name: Optional[str]=None, node_type: Optional[Union[str, NodeType]]=None, deploy_to: Optional[str]=None, node_side_type: Optional[str]=None, port: Optional[Union[int, str]]=None, processes: int=1, local_db: bool=False, dev_mode: bool=False, cmd: bool=False, reset: bool=False, tail: bool=False, host: Optional[str]='0.0.0.0', tag: Optional[str]='latest', verbose: bool=False, render: bool=False, enable_warnings: bool=False) -> Optional[NodeHandle]:
        if False:
            for i in range(10):
                print('nop')
        if dev_mode is True:
            os.environ['DEV_MODE'] = 'True'
        if node_type == 'python':
            node_type = NodeType.DOMAIN
            if deploy_to is None:
                deploy_to = 'python'
        dev_mode = str_to_bool(os.environ.get('DEV_MODE', f'{dev_mode}'))
        node_type_enum: Optional[NodeType] = get_node_type(node_type=node_type)
        if not node_type_enum:
            return None
        node_side_type_enum = NodeSideType.HIGH_SIDE if node_side_type is None else NodeSideType(node_side_type)
        deployment_type_enum: Optional[DeploymentType] = get_deployment_type(deployment_type=deploy_to)
        if not deployment_type_enum:
            return None
        if deployment_type_enum == DeploymentType.PYTHON:
            return deploy_to_python(node_type_enum=node_type_enum, deployment_type_enum=deployment_type_enum, port=port, name=name, host=host, reset=reset, tail=tail, dev_mode=dev_mode, processes=processes, local_db=local_db, node_side_type=node_side_type_enum, enable_warnings=enable_warnings)
        elif deployment_type_enum == DeploymentType.K8S:
            return deploy_to_k8s(node_type_enum=node_type_enum, deployment_type_enum=deployment_type_enum, name=name, node_side_type=node_side_type_enum)
        elif deployment_type_enum == DeploymentType.CONTAINER_STACK or deployment_type_enum == DeploymentType.SINGLE_CONTAINER:
            return deploy_to_container(node_type_enum=node_type_enum, deployment_type_enum=deployment_type_enum, reset=reset, cmd=cmd, tail=tail, verbose=verbose, tag=tag, render=render, dev_mode=dev_mode, port=port, name=name, node_side_type=node_side_type_enum, enable_warnings=enable_warnings)
        elif deployment_type_enum == DeploymentType.PODMAN:
            return deploy_to_podman(node_type_enum=node_type_enum, deployment_type_enum=deployment_type_enum, name=name, node_side_type=node_side_type_enum)
        else:
            print(f'deployment_type: {deployment_type_enum} is not supported')
            return None

    @staticmethod
    def land(name: str, deployment_type: Union[str, DeploymentType], reset: bool=False) -> None:
        if False:
            while True:
                i = 10
        deployment_type_enum = DeploymentType(deployment_type)
        Orchestra.shutdown(name=name, deployment_type_enum=deployment_type_enum)
        if reset:
            Orchestra.reset(name, deployment_type_enum=deployment_type_enum)

    @staticmethod
    def shutdown(name: str, deployment_type_enum: DeploymentType, reset: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        if deployment_type_enum != DeploymentType.PYTHON:
            snake_name = to_snake_case(name)
            if reset:
                land_output = shell(f'hagrid land {snake_name} --force --prune-vol')
            else:
                land_output = shell(f'hagrid land {snake_name} --force')
            if 'Removed' in land_output:
                print(f' ✅ {snake_name} Container Removed')
            else:
                print(f'❌ Unable to remove container: {snake_name} :{land_output}')

    @staticmethod
    def reset(name: str, deployment_type_enum: DeploymentType) -> None:
        if False:
            for i in range(10):
                print('nop')
        if deployment_type_enum == DeploymentType.PYTHON:
            sy = get_syft_client()
            _ = sy.Worker.named(name, processes=1, reset=True)
        elif deployment_type_enum == DeploymentType.CONTAINER_STACK or deployment_type_enum == DeploymentType.SINGLE_CONTAINER:
            Orchestra.shutdown(name=name, deployment_type_enum=deployment_type_enum, reset=True)
        else:
            raise NotImplementedError(f'Reset not implemented for the deployment type:{deployment_type_enum}')