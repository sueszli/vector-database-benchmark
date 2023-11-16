"""
Module for kubeadm
:maintainer:    Alberto Planas <aplanas@suse.com>
:maturity:      new
:depends:       None
:platform:      Linux
"""
import json
import logging
import re
import salt.utils.files
from salt.exceptions import CommandExecutionError
ADMIN_CFG = '/etc/kubernetes/admin.conf'
log = logging.getLogger(__name__)
__virtualname__ = 'kubeadm'
try:
    __salt__
except NameError:
    __salt__ = {}

def _api_server_endpoint(config=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the API server endpoint\n    '
    config = config if config else ADMIN_CFG
    endpoint = None
    try:
        with salt.utils.files.fopen(config, 'r') as fp_:
            endpoint = re.search('^\\s*server: https?://(.*)$', fp_.read(), re.MULTILINE).group(1)
    except Exception:
        pass
    return endpoint

def _token(create_if_needed=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a valid bootstrap token\n    '
    tokens = token_list()
    if not tokens and create_if_needed:
        token_create(description='Token created by kubeadm salt module')
        tokens = token_list()
    return tokens[0]['token'] if tokens else None

def _discovery_token_ca_cert_hash():
    if False:
        i = 10
        return i + 15
    cmd = ['openssl', 'x509', '-pubkey', '-in', '/etc/kubernetes/pki/ca.crt', '|', 'openssl', 'rsa', '-pubin', '-outform', 'der', '2>/dev/null', '|', 'openssl', 'dgst', '-sha256', '-hex', '|', 'sed', "'s/^.* //'"]
    result = __salt__['cmd.run_all'](' '.join(cmd), python_shell=True)
    if result['retcode']:
        raise CommandExecutionError(result['stderr'])
    return 'sha256:{}'.format(result['stdout'])

def join_params(create_if_needed=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 3001\n\n    Return the parameters required for joining into the cluster\n\n    create_if_needed\n       If the token bucket is empty and this parameter is True, a new\n       token will be created.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.join_params\n       salt '*' kubeadm.join_params create_if_needed=True\n\n    "
    params = {'api-server-endpoint': _api_server_endpoint(), 'token': _token(create_if_needed), 'discovery-token-ca-cert-hash': _discovery_token_ca_cert_hash()}
    return params

def version(kubeconfig=None, rootfs=None):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 3001\n\n    Return the version of kubeadm\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.version\n\n    "
    cmd = ['kubeadm', 'version']
    parameters = [('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    cmd.extend(['--output', 'json'])
    return json.loads(__salt__['cmd.run_stdout'](cmd))

def _cmd(cmd):
    if False:
        for i in range(10):
            print('nop')
    'Utility function to run commands.'
    result = __salt__['cmd.run_all'](cmd)
    if result['retcode']:
        raise CommandExecutionError(result['stderr'])
    return result['stdout']

def token_create(token=None, config=None, description=None, groups=None, ttl=None, usages=None, kubeconfig=None, rootfs=None):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 3001\n\n    Create bootstrap tokens on the server\n\n    token\n       Token to write, if None one will be generated. The token must\n       match a regular expression, that by default is\n       [a-z0-9]{6}.[a-z0-9]{16}\n\n    config\n       Path to kubeadm configuration file\n\n    description\n       A human friendly description of how this token is used\n\n    groups\n       List of extra groups that this token will authenticate, default\n       to [\'system:bootstrappers:kubeadm:default-node-token\']\n\n    ttl\n       The duration defore the token is automatically deleted (1s, 2m,\n       3h). If set to \'0\' the token will never expire. Default value\n       is 24h0m0s\n\n    usages\n       Describes the ways in which this token can be used. The default\n       value is [\'signing\', \'authentication\']\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt \'*\' kubeadm.token_create\n       salt \'*\' kubeadm.token_create a1b2c.0123456789abcdef\n       salt \'*\' kubeadm.token_create ttl=\'6h\'\n       salt \'*\' kubeadm.token_create usages="[\'signing\']"\n\n    '
    cmd = ['kubeadm', 'token', 'create']
    if token:
        cmd.append(token)
    parameters = [('config', config), ('description', description), ('groups', groups), ('ttl', ttl), ('usages', usages), ('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            if parameter in ('groups', 'usages'):
                cmd.extend(['--{}'.format(parameter), json.dumps(value)])
            else:
                cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def token_delete(token, kubeconfig=None, rootfs=None):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 3001\n\n    Delete bootstrap tokens on the server\n\n    token\n       Token to write, if None one will be generated. The token must\n       match a regular expression, that by default is\n       [a-z0-9]{6}.[a-z0-9]{16}\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.token_delete a1b2c\n       salt '*' kubeadm.token_create a1b2c.0123456789abcdef\n\n    "
    cmd = ['kubeadm', 'token', 'delete', token]
    parameters = [('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return bool(_cmd(cmd))

def token_generate(kubeconfig=None, rootfs=None):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 3001\n\n    Generate and return a bootstrap token, but do not create it on the\n    server\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.token_generate\n\n    "
    cmd = ['kubeadm', 'token', 'generate']
    parameters = [('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def token_list(kubeconfig=None, rootfs=None):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 3001\n\n    List bootstrap tokens on the server\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.token_list\n\n    "
    cmd = ['kubeadm', 'token', 'list']
    parameters = [('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    lines = _cmd(cmd).splitlines()
    tokens = []
    if lines:
        header = lines.pop(0)
        header = [i.lower() for i in re.findall('(\\w+(?:\\s\\w+)*)', header)]
        for line in lines:
            values = re.findall('(\\S+(?:\\s\\S+)*)', line)
            if len(header) != len(values):
                log.error("Error parsing line: '%s'", line)
                continue
            tokens.append({key: value for (key, value) in zip(header, values)})
    return tokens

def alpha_certs_renew(rootfs=None):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 3001\n\n    Renews certificates for a Kubernetes cluster\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.alpha_certs_renew\n\n    "
    cmd = ['kubeadm', 'alpha', 'certs', 'renew']
    parameters = [('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def alpha_kubeconfig_user(client_name, apiserver_advertise_address=None, apiserver_bind_port=None, cert_dir=None, org=None, token=None, rootfs=None):
    if False:
        return 10
    '\n    .. versionadded:: 3001\n\n    Outputs a kubeconfig file for an additional user\n\n    client_name\n       The name of the user. It will be used as the CN if client\n       certificates are created\n\n    apiserver_advertise_address\n       The IP address the API server is accessible on\n\n    apiserver_bind_port\n       The port the API server is accessible on (default 6443)\n\n    cert_dir\n       The path where certificates are stored (default\n       "/etc/kubernetes/pki")\n\n    org\n       The organization of the client certificate\n\n    token\n       The token that show be used as the authentication mechanism for\n       this kubeconfig, instead of client certificates\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt \'*\' kubeadm.alpha_kubeconfig_user client_name=user\n\n    '
    cmd = ['kubeadm', 'alpha', 'kubeconfig', 'user', '--client-name', client_name]
    parameters = [('apiserver-advertise-address', apiserver_advertise_address), ('apiserver-bind-port', apiserver_bind_port), ('cert-dir', cert_dir), ('org', org), ('token', token), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def alpha_kubelet_config_download(kubeconfig=None, kubelet_version=None, rootfs=None):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 3001\n\n    Downloads the kubelet configuration from the cluster ConfigMap\n    kubelet-config-1.X\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    kubelet_version\n       The desired version for the kubelet\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.alpha_kubelet_config_download\n       salt '*' kubeadm.alpha_kubelet_config_download kubelet_version='1.14.0'\n\n    "
    cmd = ['kubeadm', 'alpha', 'kubelet', 'config', 'download']
    parameters = [('kubeconfig', kubeconfig), ('kubelet-version', kubelet_version), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def alpha_kubelet_config_enable_dynamic(node_name, kubeconfig=None, kubelet_version=None, rootfs=None):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 3001\n\n    Enables or updates dynamic kubelet configuration for a node\n\n    node_name\n       Name of the node that should enable the dynamic kubelet\n       configuration\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    kubelet_version\n       The desired version for the kubelet\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.alpha_kubelet_config_enable_dynamic node-1\n\n    "
    cmd = ['kubeadm', 'alpha', 'kubelet', 'config', 'enable-dynamic', '--node-name', node_name]
    parameters = [('kubeconfig', kubeconfig), ('kubelet-version', kubelet_version), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def alpha_selfhosting_pivot(cert_dir=None, config=None, kubeconfig=None, store_certs_in_secrets=False, rootfs=None):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 3001\n\n    Converts a static Pod-hosted control plane into a selt-hosted one\n\n    cert_dir\n       The path where certificates are stored (default\n       "/etc/kubernetes/pki")\n\n    config\n       Path to kubeadm configuration file\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    store_certs_in_secrets\n       Enable storing certs in secrets\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt \'*\' kubeadm.alpha_selfhost_pivot\n\n    '
    cmd = ['kubeadm', 'alpha', 'selfhosting', 'pivot', '--force']
    if store_certs_in_secrets:
        cmd.append('--store-certs-in-secrets')
    parameters = [('cert-dir', cert_dir), ('config', config), ('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def config_images_list(config=None, feature_gates=None, kubernetes_version=None, kubeconfig=None, rootfs=None):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 3001\n\n    Print a list of images kubeadm will use\n\n    config\n       Path to kubeadm configuration file\n\n    feature_gates\n       A set of key=value pairs that describe feature gates for\n       various features\n\n    kubernetes_version\n       Choose a specifig Kubernetes version for the control plane\n       (default "stable-1")\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt \'*\' kubeadm.config_images_list\n\n    '
    cmd = ['kubeadm', 'config', 'images', 'list']
    parameters = [('config', config), ('feature-gates', feature_gates), ('kubernetes-version', kubernetes_version), ('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd).splitlines()

def config_images_pull(config=None, cri_socket=None, feature_gates=None, kubernetes_version=None, kubeconfig=None, rootfs=None):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 3001\n\n    Pull images used by kubeadm\n\n    config\n       Path to kubeadm configuration file\n\n    cri_socket\n       Path to the CRI socket to connect\n\n    feature_gates\n       A set of key=value pairs that describe feature gates for\n       various features\n\n    kubernetes_version\n       Choose a specifig Kubernetes version for the control plane\n       (default "stable-1")\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt \'*\' kubeadm.config_images_pull\n\n    '
    cmd = ['kubeadm', 'config', 'images', 'pull']
    parameters = [('config', config), ('cri-socket', cri_socket), ('feature-gates', feature_gates), ('kubernetes-version', kubernetes_version), ('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    prefix = '[config/images] Pulled '
    return [line.replace(prefix, '') for line in _cmd(cmd).splitlines()]

def config_migrate(old_config, new_config=None, kubeconfig=None, rootfs=None):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 3001\n\n    Read an older version of the kubeadm configuration API types from\n    a file, and output the similar config object for the newer version\n\n    old_config\n       Path to the kubeadm config file that is usin the old API\n       version and should be converted\n\n    new_config\n       Path to the resulting equivalent kubeadm config file using the\n       new API version. If not specified the output will be returned\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.config_migrate /oldconfig.cfg\n\n    "
    cmd = ['kubeadm', 'config', 'migrate', '--old-config', old_config]
    parameters = [('new-config', new_config), ('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def config_print_init_defaults(component_configs=None, kubeconfig=None, rootfs=None):
    if False:
        return 10
    "\n    .. versionadded:: 3001\n\n    Return default init configuration, that can be used for 'kubeadm\n    init'\n\n    component_config\n       A comma-separated list for component config API object to print\n       the default values for (valid values: KubeProxyConfiguration,\n       KubeletConfiguration)\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.config_print_init_defaults\n\n    "
    cmd = ['kubeadm', 'config', 'print', 'init-defaults']
    parameters = [('component-configs', component_configs), ('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def config_print_join_defaults(component_configs=None, kubeconfig=None, rootfs=None):
    if False:
        return 10
    "\n    .. versionadded:: 3001\n\n    Return default join configuration, that can be used for 'kubeadm\n    join'\n\n    component_config\n       A comma-separated list for component config API object to print\n       the default values for (valid values: KubeProxyConfiguration,\n       KubeletConfiguration)\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.config_print_join_defaults\n\n    "
    cmd = ['kubeadm', 'config', 'print', 'join-defaults']
    parameters = [('component-configs', component_configs), ('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def config_upload_from_file(config, kubeconfig=None, rootfs=None):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 3001\n\n    Upload a configuration file to the in-cluster ConfigMap for\n    kubeadm configuration\n\n    config\n       Path to a kubeadm configuration file\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.config_upload_from_file /config.cfg\n\n    "
    cmd = ['kubeadm', 'config', 'upload', 'from-file', '--config', config]
    parameters = [('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def config_upload_from_flags(apiserver_advertise_address=None, apiserver_bind_port=None, apiserver_cert_extra_sans=None, cert_dir=None, cri_socket=None, feature_gates=None, kubernetes_version=None, node_name=None, pod_network_cidr=None, service_cidr=None, service_dns_domain=None, kubeconfig=None, rootfs=None):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 3001\n\n    Create the in-cluster configuration file for the first time using\n    flags\n\n    apiserver_advertise_address\n       The IP address the API server will advertise it\'s listening on\n\n    apiserver_bind_port\n       The port the API server is accessible on (default 6443)\n\n    apiserver_cert_extra_sans\n       Optional extra Subject Alternative Names (SANs) to use for the\n       API Server serving certificate\n\n    cert_dir\n       The path where to save and store the certificates (default\n       "/etc/kubernetes/pki")\n\n    cri_socket\n       Path to the CRI socket to connect\n\n    feature_gates\n       A set of key=value pairs that describe feature gates for\n       various features\n\n    kubernetes_version\n       Choose a specifig Kubernetes version for the control plane\n       (default "stable-1")\n\n    node_name\n       Specify the node name\n\n    pod_network_cidr\n       Specify range of IP addresses for the pod network\n\n    service_cidr\n       Use alternative range of IP address for service VIPs (default\n       "10.96.0.0/12")\n\n    service_dns_domain\n       Use alternative domain for services (default "cluster.local")\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt \'*\' kubeadm.config_upload_from_flags\n\n    '
    cmd = ['kubeadm', 'config', 'upload', 'from-flags']
    parameters = [('apiserver-advertise-address', apiserver_advertise_address), ('apiserver-bind-port', apiserver_bind_port), ('apiserver-cert-extra-sans', apiserver_cert_extra_sans), ('cert-dir', cert_dir), ('cri-socket', cri_socket), ('feature-gates', feature_gates), ('kubernetes-version', kubernetes_version), ('node-name', node_name), ('pod-network-cidr', pod_network_cidr), ('service-cidr', service_cidr), ('service-dns-domain', service_dns_domain), ('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def config_view(kubeconfig=None, rootfs=None):
    if False:
        return 10
    "\n    .. versionadded:: 3001\n\n    View the kubeadm configuration stored inside the cluster\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' kubeadm.config_view\n\n    "
    cmd = ['kubeadm', 'config', 'view']
    parameters = [('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def init(apiserver_advertise_address=None, apiserver_bind_port=None, apiserver_cert_extra_sans=None, cert_dir=None, certificate_key=None, control_plane_endpoint=None, config=None, cri_socket=None, experimental_upload_certs=False, upload_certs=False, feature_gates=None, ignore_preflight_errors=None, image_repository=None, kubernetes_version=None, node_name=None, pod_network_cidr=None, service_cidr=None, service_dns_domain=None, skip_certificate_key_print=False, skip_phases=None, skip_token_print=False, token=None, token_ttl=None, rootfs=None):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 3001\n\n    Command to set up the Kubernetes control plane\n\n    apiserver_advertise_address\n       The IP address the API server will advertise it\'s listening on\n\n    apiserver_bind_port\n       The port the API server is accessible on (default 6443)\n\n    apiserver_cert_extra_sans\n       Optional extra Subject Alternative Names (SANs) to use for the\n       API Server serving certificate\n\n    cert_dir\n       The path where to save and store the certificates (default\n       "/etc/kubernetes/pki")\n\n    certificate_key\n       Key used to encrypt the control-plane certificates in the\n       kubeadm-certs Secret\n\n    config\n       Path to a kubeadm configuration file\n\n    control_plane_endpoint\n       Specify a stable IP address or DNS name for the control plane\n\n    cri_socket\n       Path to the CRI socket to connect\n\n    experimental_upload_certs\n       Upload control-plane certificate to the kubeadm-certs Secret. ( kubeadm version =< 1.16 )\n\n    upload_certs\n       Upload control-plane certificate to the kubeadm-certs Secret. ( kubeadm version > 1.16 )\n\n    feature_gates\n       A set of key=value pairs that describe feature gates for\n       various features\n\n    ignore_preflight_errors\n       A list of checks whose errors will be shown as warnings\n\n    image_repository\n       Choose a container registry to pull control plane images from\n\n    kubernetes_version\n       Choose a specifig Kubernetes version for the control plane\n       (default "stable-1")\n\n    node_name\n       Specify the node name\n\n    pod_network_cidr\n       Specify range of IP addresses for the pod network\n\n    service_cidr\n       Use alternative range of IP address for service VIPs (default\n       "10.96.0.0/12")\n\n    service_dns_domain\n       Use alternative domain for services (default "cluster.local")\n\n    skip_certificate_key_print\n       Don\'t print the key used to encrypt the control-plane\n       certificates\n\n    skip_phases\n       List of phases to be skipped\n\n    skip_token_print\n       Skip printing of the default bootstrap token generated by\n       \'kubeadm init\'\n\n    token\n       The token to use for establishing bidirectional trust between\n       nodes and control-plane nodes. The token must match a regular\n       expression, that by default is [a-z0-9]{6}.[a-z0-9]{16}\n\n    token_ttl\n       The duration defore the token is automatically deleted (1s, 2m,\n       3h). If set to \'0\' the token will never expire. Default value\n       is 24h0m0s\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt \'*\' kubeadm.init pod_network_cidr=\'10.244.0.0/16\'\n\n    '
    cmd = ['kubeadm', 'init']
    if experimental_upload_certs:
        cmd.append('--experimental-upload-certs')
    if upload_certs:
        cmd.append('--upload-certs')
    if skip_certificate_key_print:
        cmd.append('--skip-certificate-key-print')
    if skip_token_print:
        cmd.append('--skip-token-print')
    parameters = [('apiserver-advertise-address', apiserver_advertise_address), ('apiserver-bind-port', apiserver_bind_port), ('apiserver-cert-extra-sans', apiserver_cert_extra_sans), ('cert-dir', cert_dir), ('certificate-key', certificate_key), ('config', config), ('control-plane-endpoint', control_plane_endpoint), ('cri-socket', cri_socket), ('feature-gates', feature_gates), ('ignore-preflight-errors', ignore_preflight_errors), ('image-repository', image_repository), ('kubernetes-version', kubernetes_version), ('node-name', node_name), ('pod-network-cidr', pod_network_cidr), ('service-cidr', service_cidr), ('service-dns-domain', service_dns_domain), ('skip-phases', skip_phases), ('token', token), ('token-ttl', token_ttl), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def join(api_server_endpoint=None, apiserver_advertise_address=None, apiserver_bind_port=None, certificate_key=None, config=None, cri_socket=None, discovery_file=None, discovery_token=None, discovery_token_ca_cert_hash=None, discovery_token_unsafe_skip_ca_verification=False, experimental_control_plane=False, control_plane=False, ignore_preflight_errors=None, node_name=None, skip_phases=None, tls_bootstrap_token=None, token=None, rootfs=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 3001\n\n    Command to join to an existing cluster\n\n    api_server_endpoint\n       IP address or domain name and port of the API Server\n\n    apiserver_advertise_address\n       If the node should host a new control plane instance, the IP\n       address the API Server will advertise it\'s listening on\n\n    apiserver_bind_port\n       If the node should host a new control plane instance, the port\n       the API Server to bind to (default 6443)\n\n    certificate_key\n       Use this key to decrypt the certificate secrets uploaded by\n       init\n\n    config\n       Path to a kubeadm configuration file\n\n    cri_socket\n       Path to the CRI socket to connect\n\n    discovery_file\n       For file-based discovery, a file or URL from which to load\n       cluster information\n\n    discovery_token\n       For token-based discovery, the token used to validate cluster\n       information fetched from the API Server\n\n    discovery_token_ca_cert_hash\n       For token-based discovery, validate that the root CA public key\n       matches this hash (format: "<type>:<value>")\n\n    discovery_token_unsafe_skip_ca_verification\n       For token-based discovery, allow joining without\n       \'discovery-token-ca-cert-hash\' pinning\n\n    experimental_control_plane\n       Create a new control plane instance on this node (kubeadm version =< 1.16)\n\n    control_plane\n       Create a new control plane instance on this node (kubeadm version > 1.16)\n\n    ignore_preflight_errors\n       A list of checks whose errors will be shown as warnings\n\n    node_name\n       Specify the node name\n\n    skip_phases\n       List of phases to be skipped\n\n    tls_bootstrap_token\n       Specify the token used to temporarily authenticate with the\n       Kubernetes Control Plane while joining the node\n\n    token\n       Use this token for both discovery-token and tls-bootstrap-token\n       when those values are not provided\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt \'*\' kubeadm.join 10.160.65.165:6443 token=\'token\'\n\n    '
    cmd = ['kubeadm', 'join']
    if api_server_endpoint:
        cmd.append(api_server_endpoint)
    if discovery_token_unsafe_skip_ca_verification:
        cmd.append('--discovery-token-unsafe-skip-ca-verification')
    if experimental_control_plane:
        cmd.append('--experimental-control-plane')
    if control_plane:
        cmd.append('--control-plane')
    parameters = [('apiserver-advertise-address', apiserver_advertise_address), ('apiserver-bind-port', apiserver_bind_port), ('certificate-key', certificate_key), ('config', config), ('cri-socket', cri_socket), ('discovery-file', discovery_file), ('discovery-token', discovery_token), ('discovery-token-ca-cert-hash', discovery_token_ca_cert_hash), ('ignore-preflight-errors', ignore_preflight_errors), ('node-name', node_name), ('skip-phases', skip_phases), ('tls-bootstrap-token', tls_bootstrap_token), ('token', token), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)

def reset(cert_dir=None, cri_socket=None, ignore_preflight_errors=None, kubeconfig=None, rootfs=None):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 3001\n\n    Revert any changes made to this host by \'kubeadm init\' or \'kubeadm\n    join\'\n\n    cert_dir\n       The path to the directory where the certificates are stored\n       (default "/etc/kubernetes/pki")\n\n    cri_socket\n       Path to the CRI socket to connect\n\n    ignore_preflight_errors\n       A list of checks whose errors will be shown as warnings\n\n    kubeconfig\n       The kubeconfig file to use when talking to the cluster. The\n       default values in /etc/kubernetes/admin.conf\n\n    rootfs\n       The path to the real host root filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt \'*\' kubeadm.join 10.160.65.165:6443 token=\'token\'\n\n    '
    cmd = ['kubeadm', 'reset', '--force']
    parameters = [('cert-dir', cert_dir), ('cri-socket', cri_socket), ('ignore-preflight-errors', ignore_preflight_errors), ('kubeconfig', kubeconfig), ('rootfs', rootfs)]
    for (parameter, value) in parameters:
        if value:
            cmd.extend(['--{}'.format(parameter), str(value)])
    return _cmd(cmd)