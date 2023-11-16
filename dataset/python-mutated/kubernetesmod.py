"""
Module for handling kubernetes calls.

:optdepends:    - kubernetes Python client < 4.0
                - PyYAML < 6.0
:configuration: The k8s API settings are provided either in a pillar, in
    the minion's config file, or in master's config file::

        kubernetes.kubeconfig: '/path/to/kubeconfig'
        kubernetes.kubeconfig-data: '<base64 encoded kubeconfig content'
        kubernetes.context: 'context'

These settings can be overridden by adding `context and `kubeconfig` or
`kubeconfig_data` parameters when calling a function.

The data format for `kubernetes.kubeconfig-data` value is the content of
`kubeconfig` base64 encoded in one line.

Only `kubeconfig` or `kubeconfig-data` should be provided. In case both are
provided `kubeconfig` entry is preferred.

CLI Example:

.. code-block:: bash

    salt '*' kubernetes.nodes kubeconfig=/etc/salt/k8s/kubeconfig context=minikube

.. versionadded:: 2017.7.0
.. versionchanged:: 2019.2.0

.. warning::

    Configuration options changed in 2019.2.0. The following configuration options have been removed:

    - kubernetes.user
    - kubernetes.password
    - kubernetes.api_url
    - kubernetes.certificate-authority-data/file
    - kubernetes.client-certificate-data/file
    - kubernetes.client-key-data/file

    Please use now:

    - kubernetes.kubeconfig or kubernetes.kubeconfig-data
    - kubernetes.context

"""
import base64
import errno
import logging
import os.path
import signal
import sys
import tempfile
import time
from contextlib import contextmanager
import salt.utils.files
import salt.utils.platform
import salt.utils.templates
import salt.utils.yaml
from salt.exceptions import CommandExecutionError, TimeoutError
try:
    import kubernetes
    import kubernetes.client
    from kubernetes.client.rest import ApiException
    from urllib3.exceptions import HTTPError
    try:
        from kubernetes.client import V1beta1Deployment as AppsV1beta1Deployment
        from kubernetes.client import V1beta1DeploymentSpec as AppsV1beta1DeploymentSpec
    except ImportError:
        from kubernetes.client import AppsV1beta1Deployment, AppsV1beta1DeploymentSpec
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
log = logging.getLogger(__name__)
__virtualname__ = 'kubernetes'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Check dependencies\n    '
    if HAS_LIBS:
        return __virtualname__
    return (False, 'python kubernetes library not found')
if not salt.utils.platform.is_windows():

    @contextmanager
    def _time_limit(seconds):
        if False:
            for i in range(10):
                print('nop')

        def signal_handler(signum, frame):
            if False:
                while True:
                    i = 10
            raise TimeoutError
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    POLLING_TIME_LIMIT = 30

def _setup_conn_old(**kwargs):
    if False:
        while True:
            i = 10
    '\n    Setup kubernetes API connection singleton the old way\n    '
    host = __salt__['config.option']('kubernetes.api_url', 'http://localhost:8080')
    username = __salt__['config.option']('kubernetes.user')
    password = __salt__['config.option']('kubernetes.password')
    ca_cert = __salt__['config.option']('kubernetes.certificate-authority-data')
    client_cert = __salt__['config.option']('kubernetes.client-certificate-data')
    client_key = __salt__['config.option']('kubernetes.client-key-data')
    ca_cert_file = __salt__['config.option']('kubernetes.certificate-authority-file')
    client_cert_file = __salt__['config.option']('kubernetes.client-certificate-file')
    client_key_file = __salt__['config.option']('kubernetes.client-key-file')
    if 'api_url' in kwargs:
        host = kwargs.get('api_url')
    if 'api_user' in kwargs:
        username = kwargs.get('api_user')
    if 'api_password' in kwargs:
        password = kwargs.get('api_password')
    if 'api_certificate_authority_file' in kwargs:
        ca_cert_file = kwargs.get('api_certificate_authority_file')
    if 'api_client_certificate_file' in kwargs:
        client_cert_file = kwargs.get('api_client_certificate_file')
    if 'api_client_key_file' in kwargs:
        client_key_file = kwargs.get('api_client_key_file')
    if kubernetes.client.configuration.host != host or kubernetes.client.configuration.user != username or kubernetes.client.configuration.password != password:
        kubernetes.client.configuration.__init__()
    kubernetes.client.configuration.host = host
    kubernetes.client.configuration.user = username
    kubernetes.client.configuration.passwd = password
    if ca_cert_file:
        kubernetes.client.configuration.ssl_ca_cert = ca_cert_file
    elif ca_cert:
        with tempfile.NamedTemporaryFile(prefix='salt-kube-', delete=False) as ca:
            ca.write(base64.b64decode(ca_cert))
            kubernetes.client.configuration.ssl_ca_cert = ca.name
    else:
        kubernetes.client.configuration.ssl_ca_cert = None
    if client_cert_file:
        kubernetes.client.configuration.cert_file = client_cert_file
    elif client_cert:
        with tempfile.NamedTemporaryFile(prefix='salt-kube-', delete=False) as c:
            c.write(base64.b64decode(client_cert))
            kubernetes.client.configuration.cert_file = c.name
    else:
        kubernetes.client.configuration.cert_file = None
    if client_key_file:
        kubernetes.client.configuration.key_file = client_key_file
    elif client_key:
        with tempfile.NamedTemporaryFile(prefix='salt-kube-', delete=False) as k:
            k.write(base64.b64decode(client_key))
            kubernetes.client.configuration.key_file = k.name
    else:
        kubernetes.client.configuration.key_file = None
    return {}

def _setup_conn(**kwargs):
    if False:
        while True:
            i = 10
    '\n    Setup kubernetes API connection singleton\n    '
    kubeconfig = kwargs.get('kubeconfig') or __salt__['config.option']('kubernetes.kubeconfig')
    kubeconfig_data = kwargs.get('kubeconfig_data') or __salt__['config.option']('kubernetes.kubeconfig-data')
    context = kwargs.get('context') or __salt__['config.option']('kubernetes.context')
    if kubeconfig_data and (not kubeconfig) or (kubeconfig_data and kwargs.get('kubeconfig_data')):
        with tempfile.NamedTemporaryFile(prefix='salt-kubeconfig-', delete=False) as kcfg:
            kcfg.write(base64.b64decode(kubeconfig_data))
            kubeconfig = kcfg.name
    if not (kubeconfig and context):
        if kwargs.get('api_url') or __salt__['config.option']('kubernetes.api_url'):
            try:
                return _setup_conn_old(**kwargs)
            except Exception:
                raise CommandExecutionError('Old style kubernetes configuration is only supported up to python-kubernetes 2.0.0')
        else:
            raise CommandExecutionError("Invalid kubernetes configuration. Parameter 'kubeconfig' and 'context' are required.")
    kubernetes.config.load_kube_config(config_file=kubeconfig, context=context)
    return {'kubeconfig': kubeconfig, 'context': context}

def _cleanup_old(**kwargs):
    if False:
        print('Hello World!')
    try:
        ca = kubernetes.client.configuration.ssl_ca_cert
        cert = kubernetes.client.configuration.cert_file
        key = kubernetes.client.configuration.key_file
        if cert and os.path.exists(cert) and os.path.basename(cert).startswith('salt-kube-'):
            salt.utils.files.safe_rm(cert)
        if key and os.path.exists(key) and os.path.basename(key).startswith('salt-kube-'):
            salt.utils.files.safe_rm(key)
        if ca and os.path.exists(ca) and os.path.basename(ca).startswith('salt-kube-'):
            salt.utils.files.safe_rm(ca)
    except Exception:
        pass

def _cleanup(**kwargs):
    if False:
        print('Hello World!')
    if not kwargs:
        return _cleanup_old(**kwargs)
    if 'kubeconfig' in kwargs:
        kubeconfig = kwargs.get('kubeconfig')
        if kubeconfig and os.path.basename(kubeconfig).startswith('salt-kubeconfig-'):
            try:
                os.unlink(kubeconfig)
            except OSError as err:
                if err.errno != errno.ENOENT:
                    log.exception(err)

def ping(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Checks connections with the kubernetes API server.\n    Returns True if the connection can be established, False otherwise.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.ping\n    "
    status = True
    try:
        nodes(**kwargs)
    except CommandExecutionError:
        status = False
    return status

def nodes(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Return the names of the nodes composing the kubernetes cluster\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.nodes\n        salt '*' kubernetes.nodes kubeconfig=/etc/salt/k8s/kubeconfig context=minikube\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.list_node()
        return [k8s_node['metadata']['name'] for k8s_node in api_response.to_dict().get('items')]
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->list_node')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def node(name, **kwargs):
    if False:
        return 10
    "\n    Return the details of the node identified by the specified name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.node name='minikube'\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.list_node()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->list_node')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)
    for k8s_node in api_response.items:
        if k8s_node.metadata.name == name:
            return k8s_node.to_dict()
    return None

def node_labels(name, **kwargs):
    if False:
        return 10
    '\n    Return the labels of the node identified by the specified name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' kubernetes.node_labels name="minikube"\n    '
    match = node(name, **kwargs)
    if match is not None:
        return match['metadata']['labels']
    return {}

def node_add_label(node_name, label_name, label_value, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Set the value of the label identified by `label_name` to `label_value` on\n    the node identified by the name `node_name`.\n    Creates the label if not present.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' kubernetes.node_add_label node_name="minikube"             label_name="foo" label_value="bar"\n    '
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        body = {'metadata': {'labels': {label_name: label_value}}}
        api_response = api_instance.patch_node(node_name, body)
        return api_response
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->patch_node')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)
    return None

def node_remove_label(node_name, label_name, **kwargs):
    if False:
        print('Hello World!')
    '\n    Removes the label identified by `label_name` from\n    the node identified by the name `node_name`.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' kubernetes.node_remove_label node_name="minikube"             label_name="foo"\n    '
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        body = {'metadata': {'labels': {label_name: None}}}
        api_response = api_instance.patch_node(node_name, body)
        return api_response
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->patch_node')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)
    return None

def namespaces(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return the names of the available namespaces\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.namespaces\n        salt '*' kubernetes.namespaces kubeconfig=/etc/salt/k8s/kubeconfig context=minikube\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.list_namespace()
        return [nms['metadata']['name'] for nms in api_response.to_dict().get('items')]
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->list_namespace')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def deployments(namespace='default', **kwargs):
    if False:
        return 10
    "\n    Return a list of kubernetes deployments defined in the namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.deployments\n        salt '*' kubernetes.deployments namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.ExtensionsV1beta1Api()
        api_response = api_instance.list_namespaced_deployment(namespace)
        return [dep['metadata']['name'] for dep in api_response.to_dict().get('items')]
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling ExtensionsV1beta1Api->list_namespaced_deployment')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def services(namespace='default', **kwargs):
    if False:
        return 10
    "\n    Return a list of kubernetes services defined in the namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.services\n        salt '*' kubernetes.services namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.list_namespaced_service(namespace)
        return [srv['metadata']['name'] for srv in api_response.to_dict().get('items')]
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->list_namespaced_service')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def pods(namespace='default', **kwargs):
    if False:
        return 10
    "\n    Return a list of kubernetes pods defined in the namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.pods\n        salt '*' kubernetes.pods namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.list_namespaced_pod(namespace)
        return [pod['metadata']['name'] for pod in api_response.to_dict().get('items')]
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->list_namespaced_pod')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def secrets(namespace='default', **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return a list of kubernetes secrets defined in the namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.secrets\n        salt '*' kubernetes.secrets namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.list_namespaced_secret(namespace)
        return [secret['metadata']['name'] for secret in api_response.to_dict().get('items')]
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->list_namespaced_secret')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def configmaps(namespace='default', **kwargs):
    if False:
        print('Hello World!')
    "\n    Return a list of kubernetes configmaps defined in the namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.configmaps\n        salt '*' kubernetes.configmaps namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.list_namespaced_config_map(namespace)
        return [secret['metadata']['name'] for secret in api_response.to_dict().get('items')]
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->list_namespaced_config_map')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def show_deployment(name, namespace='default', **kwargs):
    if False:
        while True:
            i = 10
    "\n    Return the kubernetes deployment defined by name and namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.show_deployment my-nginx default\n        salt '*' kubernetes.show_deployment name=my-nginx namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.ExtensionsV1beta1Api()
        api_response = api_instance.read_namespaced_deployment(name, namespace)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling ExtensionsV1beta1Api->read_namespaced_deployment')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def show_service(name, namespace='default', **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return the kubernetes service defined by name and namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.show_service my-nginx default\n        salt '*' kubernetes.show_service name=my-nginx namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.read_namespaced_service(name, namespace)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->read_namespaced_service')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def show_pod(name, namespace='default', **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return POD information for a given pod name defined in the namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.show_pod guestbook-708336848-fqr2x\n        salt '*' kubernetes.show_pod guestbook-708336848-fqr2x namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.read_namespaced_pod(name, namespace)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->read_namespaced_pod')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def show_namespace(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return information for a given namespace defined by the specified name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.show_namespace kube-system\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.read_namespace(name)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->read_namespace')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def show_secret(name, namespace='default', decode=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the kubernetes secret defined by name and namespace.\n    The secrets can be decoded if specified by the user. Warning: this has\n    security implications.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.show_secret confidential default\n        salt '*' kubernetes.show_secret name=confidential namespace=default\n        salt '*' kubernetes.show_secret name=confidential decode=True\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.read_namespaced_secret(name, namespace)
        if api_response.data and (decode or decode == 'True'):
            for key in api_response.data:
                value = api_response.data[key]
                api_response.data[key] = base64.b64decode(value)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->read_namespaced_secret')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def show_configmap(name, namespace='default', **kwargs):
    if False:
        print('Hello World!')
    "\n    Return the kubernetes configmap defined by name and namespace.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.show_configmap game-config default\n        salt '*' kubernetes.show_configmap name=game-config namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.read_namespaced_config_map(name, namespace)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->read_namespaced_config_map')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def delete_deployment(name, namespace='default', **kwargs):
    if False:
        return 10
    "\n    Deletes the kubernetes deployment defined by name and namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.delete_deployment my-nginx\n        salt '*' kubernetes.delete_deployment name=my-nginx namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    body = kubernetes.client.V1DeleteOptions(orphan_dependents=True)
    try:
        api_instance = kubernetes.client.ExtensionsV1beta1Api()
        api_response = api_instance.delete_namespaced_deployment(name=name, namespace=namespace, body=body)
        mutable_api_response = api_response.to_dict()
        if not salt.utils.platform.is_windows():
            try:
                with _time_limit(POLLING_TIME_LIMIT):
                    while show_deployment(name, namespace) is not None:
                        time.sleep(1)
                    else:
                        mutable_api_response['code'] = 200
            except TimeoutError:
                pass
        else:
            for i in range(60):
                if show_deployment(name, namespace) is None:
                    mutable_api_response['code'] = 200
                    break
                else:
                    time.sleep(1)
        if mutable_api_response['code'] != 200:
            log.warning("Reached polling time limit. Deployment is not yet deleted, but we are backing off. Sorry, but you'll have to check manually.")
        return mutable_api_response
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling ExtensionsV1beta1Api->delete_namespaced_deployment')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def delete_service(name, namespace='default', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Deletes the kubernetes service defined by name and namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.delete_service my-nginx default\n        salt '*' kubernetes.delete_service name=my-nginx namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.delete_namespaced_service(name=name, namespace=namespace)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->delete_namespaced_service')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def delete_pod(name, namespace='default', **kwargs):
    if False:
        print('Hello World!')
    "\n    Deletes the kubernetes pod defined by name and namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.delete_pod guestbook-708336848-5nl8c default\n        salt '*' kubernetes.delete_pod name=guestbook-708336848-5nl8c namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    body = kubernetes.client.V1DeleteOptions(orphan_dependents=True)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.delete_namespaced_pod(name=name, namespace=namespace, body=body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->delete_namespaced_pod')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def delete_namespace(name, **kwargs):
    if False:
        print('Hello World!')
    "\n    Deletes the kubernetes namespace defined by name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.delete_namespace salt\n        salt '*' kubernetes.delete_namespace name=salt\n    "
    cfg = _setup_conn(**kwargs)
    body = kubernetes.client.V1DeleteOptions(orphan_dependents=True)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.delete_namespace(name=name, body=body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->delete_namespace')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def delete_secret(name, namespace='default', **kwargs):
    if False:
        return 10
    "\n    Deletes the kubernetes secret defined by name and namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.delete_secret confidential default\n        salt '*' kubernetes.delete_secret name=confidential namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    body = kubernetes.client.V1DeleteOptions(orphan_dependents=True)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.delete_namespaced_secret(name=name, namespace=namespace, body=body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->delete_namespaced_secret')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def delete_configmap(name, namespace='default', **kwargs):
    if False:
        while True:
            i = 10
    "\n    Deletes the kubernetes configmap defined by name and namespace\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.delete_configmap settings default\n        salt '*' kubernetes.delete_configmap name=settings namespace=default\n    "
    cfg = _setup_conn(**kwargs)
    body = kubernetes.client.V1DeleteOptions(orphan_dependents=True)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.delete_namespaced_config_map(name=name, namespace=namespace, body=body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->delete_namespaced_config_map')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def create_deployment(name, namespace, metadata, spec, source, template, saltenv, **kwargs):
    if False:
        print('Hello World!')
    '\n    Creates the kubernetes deployment as defined by the user.\n    '
    body = __create_object_body(kind='Deployment', obj_class=AppsV1beta1Deployment, spec_creator=__dict_to_deployment_spec, name=name, namespace=namespace, metadata=metadata, spec=spec, source=source, template=template, saltenv=saltenv)
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.ExtensionsV1beta1Api()
        api_response = api_instance.create_namespaced_deployment(namespace, body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling ExtensionsV1beta1Api->create_namespaced_deployment')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def create_pod(name, namespace, metadata, spec, source, template, saltenv, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Creates the kubernetes deployment as defined by the user.\n    '
    body = __create_object_body(kind='Pod', obj_class=kubernetes.client.V1Pod, spec_creator=__dict_to_pod_spec, name=name, namespace=namespace, metadata=metadata, spec=spec, source=source, template=template, saltenv=saltenv)
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.create_namespaced_pod(namespace, body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->create_namespaced_pod')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def create_service(name, namespace, metadata, spec, source, template, saltenv, **kwargs):
    if False:
        print('Hello World!')
    '\n    Creates the kubernetes service as defined by the user.\n    '
    body = __create_object_body(kind='Service', obj_class=kubernetes.client.V1Service, spec_creator=__dict_to_service_spec, name=name, namespace=namespace, metadata=metadata, spec=spec, source=source, template=template, saltenv=saltenv)
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.create_namespaced_service(namespace, body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->create_namespaced_service')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def create_secret(name, namespace='default', data=None, source=None, template=None, saltenv='base', **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Creates the kubernetes secret as defined by the user.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'minion1\' kubernetes.create_secret             passwords default \'{"db": "letmein"}\'\n\n        salt \'minion2\' kubernetes.create_secret             name=passwords namespace=default data=\'{"db": "letmein"}\'\n    '
    if source:
        data = __read_and_render_yaml_file(source, template, saltenv)
    elif data is None:
        data = {}
    data = __enforce_only_strings_dict(data)
    for key in data:
        data[key] = base64.b64encode(data[key])
    body = kubernetes.client.V1Secret(metadata=__dict_to_object_meta(name, namespace, {}), data=data)
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.create_namespaced_secret(namespace, body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->create_namespaced_secret')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def create_configmap(name, namespace, data, source=None, template=None, saltenv='base', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates the kubernetes configmap as defined by the user.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'minion1\' kubernetes.create_configmap             settings default \'{"example.conf": "# example file"}\'\n\n        salt \'minion2\' kubernetes.create_configmap             name=settings namespace=default data=\'{"example.conf": "# example file"}\'\n    '
    if source:
        data = __read_and_render_yaml_file(source, template, saltenv)
    elif data is None:
        data = {}
    data = __enforce_only_strings_dict(data)
    body = kubernetes.client.V1ConfigMap(metadata=__dict_to_object_meta(name, namespace, {}), data=data)
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.create_namespaced_config_map(namespace, body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->create_namespaced_config_map')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def create_namespace(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Creates a namespace with the specified name.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kubernetes.create_namespace salt\n        salt '*' kubernetes.create_namespace name=salt\n    "
    meta_obj = kubernetes.client.V1ObjectMeta(name=name)
    body = kubernetes.client.V1Namespace(metadata=meta_obj)
    body.metadata.name = name
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.create_namespace(body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->create_namespace')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def replace_deployment(name, metadata, spec, source, template, saltenv, namespace='default', **kwargs):
    if False:
        return 10
    '\n    Replaces an existing deployment with a new one defined by name and\n    namespace, having the specificed metadata and spec.\n    '
    body = __create_object_body(kind='Deployment', obj_class=AppsV1beta1Deployment, spec_creator=__dict_to_deployment_spec, name=name, namespace=namespace, metadata=metadata, spec=spec, source=source, template=template, saltenv=saltenv)
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.ExtensionsV1beta1Api()
        api_response = api_instance.replace_namespaced_deployment(name, namespace, body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling ExtensionsV1beta1Api->replace_namespaced_deployment')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def replace_service(name, metadata, spec, source, template, old_service, saltenv, namespace='default', **kwargs):
    if False:
        return 10
    '\n    Replaces an existing service with a new one defined by name and namespace,\n    having the specificed metadata and spec.\n    '
    body = __create_object_body(kind='Service', obj_class=kubernetes.client.V1Service, spec_creator=__dict_to_service_spec, name=name, namespace=namespace, metadata=metadata, spec=spec, source=source, template=template, saltenv=saltenv)
    body.spec.cluster_ip = old_service['spec']['cluster_ip']
    body.metadata.resource_version = old_service['metadata']['resource_version']
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.replace_namespaced_service(name, namespace, body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->replace_namespaced_service')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def replace_secret(name, data, source=None, template=None, saltenv='base', namespace='default', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Replaces an existing secret with a new one defined by name and namespace,\n    having the specificed data.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'minion1\' kubernetes.replace_secret             name=passwords data=\'{"db": "letmein"}\'\n\n        salt \'minion2\' kubernetes.replace_secret             name=passwords namespace=saltstack data=\'{"db": "passw0rd"}\'\n    '
    if source:
        data = __read_and_render_yaml_file(source, template, saltenv)
    elif data is None:
        data = {}
    data = __enforce_only_strings_dict(data)
    for key in data:
        data[key] = base64.b64encode(data[key])
    body = kubernetes.client.V1Secret(metadata=__dict_to_object_meta(name, namespace, {}), data=data)
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.replace_namespaced_secret(name, namespace, body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->replace_namespaced_secret')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def replace_configmap(name, data, source=None, template=None, saltenv='base', namespace='default', **kwargs):
    if False:
        while True:
            i = 10
    '\n    Replaces an existing configmap with a new one defined by name and\n    namespace with the specified data.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'minion1\' kubernetes.replace_configmap             settings default \'{"example.conf": "# example file"}\'\n\n        salt \'minion2\' kubernetes.replace_configmap             name=settings namespace=default data=\'{"example.conf": "# example file"}\'\n    '
    if source:
        data = __read_and_render_yaml_file(source, template, saltenv)
    data = __enforce_only_strings_dict(data)
    body = kubernetes.client.V1ConfigMap(metadata=__dict_to_object_meta(name, namespace, {}), data=data)
    cfg = _setup_conn(**kwargs)
    try:
        api_instance = kubernetes.client.CoreV1Api()
        api_response = api_instance.replace_namespaced_config_map(name, namespace, body)
        return api_response.to_dict()
    except (ApiException, HTTPError) as exc:
        if isinstance(exc, ApiException) and exc.status == 404:
            return None
        else:
            log.exception('Exception when calling CoreV1Api->replace_namespaced_configmap')
            raise CommandExecutionError(exc)
    finally:
        _cleanup(**cfg)

def __create_object_body(kind, obj_class, spec_creator, name, namespace, metadata, spec, source, template, saltenv):
    if False:
        i = 10
        return i + 15
    '\n    Create a Kubernetes Object body instance.\n    '
    if source:
        src_obj = __read_and_render_yaml_file(source, template, saltenv)
        if not isinstance(src_obj, dict) or 'kind' not in src_obj or src_obj['kind'] != kind:
            raise CommandExecutionError('The source file should define only a {} object'.format(kind))
        if 'metadata' in src_obj:
            metadata = src_obj['metadata']
        if 'spec' in src_obj:
            spec = src_obj['spec']
    return obj_class(metadata=__dict_to_object_meta(name, namespace, metadata), spec=spec_creator(spec))

def __read_and_render_yaml_file(source, template, saltenv):
    if False:
        for i in range(10):
            print('nop')
    '\n    Read a yaml file and, if needed, renders that using the specifieds\n    templating. Returns the python objects defined inside of the file.\n    '
    sfn = __salt__['cp.cache_file'](source, saltenv)
    if not sfn:
        raise CommandExecutionError("Source file '{}' not found".format(source))
    with salt.utils.files.fopen(sfn, 'r') as src:
        contents = src.read()
        if template:
            if template in salt.utils.templates.TEMPLATE_REGISTRY:
                data = salt.utils.templates.TEMPLATE_REGISTRY[template](contents, from_str=True, to_str=True, saltenv=saltenv, grains=__grains__, pillar=__pillar__, salt=__salt__, opts=__opts__)
                if not data['result']:
                    raise CommandExecutionError('Failed to render file path with error: {}'.format(data['data']))
                contents = data['data'].encode('utf-8')
            else:
                raise CommandExecutionError('Unknown template specified: {}'.format(template))
        return salt.utils.yaml.safe_load(contents)

def __dict_to_object_meta(name, namespace, metadata):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts a dictionary into kubernetes ObjectMetaV1 instance.\n    '
    meta_obj = kubernetes.client.V1ObjectMeta()
    meta_obj.namespace = namespace
    if 'annotations' not in metadata:
        metadata['annotations'] = {}
    if 'kubernetes.io/change-cause' not in metadata['annotations']:
        metadata['annotations']['kubernetes.io/change-cause'] = ' '.join(sys.argv)
    for (key, value) in metadata.items():
        if hasattr(meta_obj, key):
            setattr(meta_obj, key, value)
    if meta_obj.name != name:
        log.warning('The object already has a name attribute, overwriting it with the one defined inside of salt')
        meta_obj.name = name
    return meta_obj

def __dict_to_deployment_spec(spec):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts a dictionary into kubernetes AppsV1beta1DeploymentSpec instance.\n    '
    spec_obj = AppsV1beta1DeploymentSpec(template=spec.get('template', ''))
    for (key, value) in spec.items():
        if hasattr(spec_obj, key):
            setattr(spec_obj, key, value)
    return spec_obj

def __dict_to_pod_spec(spec):
    if False:
        print('Hello World!')
    '\n    Converts a dictionary into kubernetes V1PodSpec instance.\n    '
    spec_obj = kubernetes.client.V1PodSpec()
    for (key, value) in spec.items():
        if hasattr(spec_obj, key):
            setattr(spec_obj, key, value)
    return spec_obj

def __dict_to_service_spec(spec):
    if False:
        i = 10
        return i + 15
    '\n    Converts a dictionary into kubernetes V1ServiceSpec instance.\n    '
    spec_obj = kubernetes.client.V1ServiceSpec()
    for (key, value) in spec.items():
        if key == 'ports':
            spec_obj.ports = []
            for port in value:
                kube_port = kubernetes.client.V1ServicePort()
                if isinstance(port, dict):
                    for (port_key, port_value) in port.items():
                        if hasattr(kube_port, port_key):
                            setattr(kube_port, port_key, port_value)
                else:
                    kube_port.port = port
                spec_obj.ports.append(kube_port)
        elif hasattr(spec_obj, key):
            setattr(spec_obj, key, value)
    return spec_obj

def __enforce_only_strings_dict(dictionary):
    if False:
        print('Hello World!')
    '\n    Returns a dictionary that has string keys and values.\n    '
    ret = {}
    for (key, value) in dictionary.items():
        ret[str(key)] = str(value)
    return ret