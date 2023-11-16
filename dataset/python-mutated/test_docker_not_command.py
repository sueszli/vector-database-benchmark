import pytest
from io import BytesIO
from thefuck.types import Command
from thefuck.rules.docker_not_command import get_new_command, match
_DOCKER_SWARM_OUTPUT = "\nUsage:\tdocker swarm COMMAND\n\nManage Swarm\n\nCommands:\n  ca          Display and rotate the root CA\n  init        Initialize a swarm\n  join        Join a swarm as a node and/or manager\n  join-token  Manage join tokens\n  leave       Leave the swarm\n  unlock      Unlock swarm\n  unlock-key  Manage the unlock key\n  update      Update the swarm\n\nRun 'docker swarm COMMAND --help' for more information on a command.\n"
_DOCKER_IMAGE_OUTPUT = "\nUsage:\tdocker image COMMAND\n\nManage images\n\nCommands:\n  build       Build an image from a Dockerfile\n  history     Show the history of an image\n  import      Import the contents from a tarball to create a filesystem image\n  inspect     Display detailed information on one or more images\n  load        Load an image from a tar archive or STDIN\n  ls          List images\n  prune       Remove unused images\n  pull        Pull an image or a repository from a registry\n  push        Push an image or a repository to a registry\n  rm          Remove one or more images\n  save        Save one or more images to a tar archive (streamed to STDOUT by default)\n  tag         Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE\n\nRun 'docker image COMMAND --help' for more information on a command.\n"

@pytest.fixture
def docker_help(mocker):
    if False:
        while True:
            i = 10
    help = b"Usage: docker [OPTIONS] COMMAND [arg...]\n\nA self-sufficient runtime for linux containers.\n\nOptions:\n\n  --api-cors-header=                   Set CORS headers in the remote API\n  -b, --bridge=                        Attach containers to a network bridge\n  --bip=                               Specify network bridge IP\n  -D, --debug=false                    Enable debug mode\n  -d, --daemon=false                   Enable daemon mode\n  --default-gateway=                   Container default gateway IPv4 address\n  --default-gateway-v6=                Container default gateway IPv6 address\n  --default-ulimit=[]                  Set default ulimits for containers\n  --dns=[]                             DNS server to use\n  --dns-search=[]                      DNS search domains to use\n  -e, --exec-driver=native             Exec driver to use\n  --exec-opt=[]                        Set exec driver options\n  --exec-root=/var/run/docker          Root of the Docker execdriver\n  --fixed-cidr=                        IPv4 subnet for fixed IPs\n  --fixed-cidr-v6=                     IPv6 subnet for fixed IPs\n  -G, --group=docker                   Group for the unix socket\n  -g, --graph=/var/lib/docker          Root of the Docker runtime\n  -H, --host=[]                        Daemon socket(s) to connect to\n  -h, --help=false                     Print usage\n  --icc=true                           Enable inter-container communication\n  --insecure-registry=[]               Enable insecure registry communication\n  --ip=0.0.0.0                         Default IP when binding container ports\n  --ip-forward=true                    Enable net.ipv4.ip_forward\n  --ip-masq=true                       Enable IP masquerading\n  --iptables=true                      Enable addition of iptables rules\n  --ipv6=false                         Enable IPv6 networking\n  -l, --log-level=info                 Set the logging level\n  --label=[]                           Set key=value labels to the daemon\n  --log-driver=json-file               Default driver for container logs\n  --log-opt=map[]                      Set log driver options\n  --mtu=0                              Set the containers network MTU\n  -p, --pidfile=/var/run/docker.pid    Path to use for daemon PID file\n  --registry-mirror=[]                 Preferred Docker registry mirror\n  -s, --storage-driver=                Storage driver to use\n  --selinux-enabled=false              Enable selinux support\n  --storage-opt=[]                     Set storage driver options\n  --tls=false                          Use TLS; implied by --tlsverify\n  --tlscacert=~/.docker/ca.pem         Trust certs signed only by this CA\n  --tlscert=~/.docker/cert.pem         Path to TLS certificate file\n  --tlskey=~/.docker/key.pem           Path to TLS key file\n  --tlsverify=false                    Use TLS and verify the remote\n  --userland-proxy=true                Use userland proxy for loopback traffic\n  -v, --version=false                  Print version information and quit\n\nCommands:\n    attach    Attach to a running container\n    build     Build an image from a Dockerfile\n    commit    Create a new image from a container's changes\n    cp        Copy files/folders from a container's filesystem to the host path\n    create    Create a new container\n    diff      Inspect changes on a container's filesystem\n    events    Get real time events from the server\n    exec      Run a command in a running container\n    export    Stream the contents of a container as a tar archive\n    history   Show the history of an image\n    images    List images\n    import    Create a new filesystem image from the contents of a tarball\n    info      Display system-wide information\n    inspect   Return low-level information on a container or image\n    kill      Kill a running container\n    load      Load an image from a tar archive\n    login     Register or log in to a Docker registry server\n    logout    Log out from a Docker registry server\n    logs      Fetch the logs of a container\n    pause     Pause all processes within a container\n    port      Lookup the public-facing port that is NAT-ed to PRIVATE_PORT\n    ps        List containers\n    pull      Pull an image or a repository from a Docker registry server\n    push      Push an image or a repository to a Docker registry server\n    rename    Rename an existing container\n    restart   Restart a running container\n    rm        Remove one or more containers\n    rmi       Remove one or more images\n    run       Run a command in a new container\n    save      Save an image to a tar archive\n    search    Search for an image on the Docker Hub\n    start     Start a stopped container\n    stats     Display a stream of a containers' resource usage statistics\n    stop      Stop a running container\n    tag       Tag an image into a repository\n    top       Lookup the running processes of a container\n    unpause   Unpause a paused container\n    version   Show the Docker version information\n    wait      Block until a container stops, then print its exit code\n\nRun 'docker COMMAND --help' for more information on a command.\n"
    mock = mocker.patch('subprocess.Popen')
    mock.return_value.stdout = BytesIO(help)
    return mock

@pytest.fixture
def docker_help_new(mocker):
    if False:
        return 10
    helptext_new = b'\nUsage:\tdocker [OPTIONS] COMMAND\n\nA self-sufficient runtime for containers\n\nOptions:\n      --config string      Location of client config files (default "/Users/ik1ne/.docker")\n  -c, --context string     Name of the context to use to connect to the daemon (overrides DOCKER_HOST env var\n                           and default context set with "docker context use")\n  -D, --debug              Enable debug mode\n  -H, --host list          Daemon socket(s) to connect to\n  -l, --log-level string   Set the logging level ("debug"|"info"|"warn"|"error"|"fatal") (default "info")\n      --tls                Use TLS; implied by --tlsverify\n      --tlscacert string   Trust certs signed only by this CA (default "/Users/ik1ne/.docker/ca.pem")\n      --tlscert string     Path to TLS certificate file (default "/Users/ik1ne/.docker/cert.pem")\n      --tlskey string      Path to TLS key file (default "/Users/ik1ne/.docker/key.pem")\n      --tlsverify          Use TLS and verify the remote\n  -v, --version            Print version information and quit\n\nManagement Commands:\n  builder     Manage builds\n  config      Manage Docker configs\n  container   Manage containers\n  context     Manage contexts\n  image       Manage images\n  network     Manage networks\n  node        Manage Swarm nodes\n  plugin      Manage plugins\n  secret      Manage Docker secrets\n  service     Manage services\n  stack       Manage Docker stacks\n  swarm       Manage Swarm\n  system      Manage Docker\n  trust       Manage trust on Docker images\n  volume      Manage volumes\n\nCommands:\n  attach      Attach local standard input, output, and error streams to a running container\n  build       Build an image from a Dockerfile\n  commit      Create a new image from a container\'s changes\n  cp          Copy files/folders between a container and the local filesystem\n  create      Create a new container\n  diff        Inspect changes to files or directories on a container\'s filesystem\n  events      Get real time events from the server\n  exec        Run a command in a running container\n  export      Export a container\'s filesystem as a tar archive\n  history     Show the history of an image\n  images      List images\n  import      Import the contents from a tarball to create a filesystem image\n  info        Display system-wide information\n  inspect     Return low-level information on Docker objects\n  kill        Kill one or more running containers\n  load        Load an image from a tar archive or STDIN\n  login       Log in to a Docker registry\n  logout      Log out from a Docker registry\n  logs        Fetch the logs of a container\n  pause       Pause all processes within one or more containers\n  port        List port mappings or a specific mapping for the container\n  ps          List containers\n  pull        Pull an image or a repository from a registry\n  push        Push an image or a repository to a registry\n  rename      Rename a container\n  restart     Restart one or more containers\n  rm          Remove one or more containers\n  rmi         Remove one or more images\n  run         Run a command in a new container\n  save        Save one or more images to a tar archive (streamed to STDOUT by default)\n  search      Search the Docker Hub for images\n  start       Start one or more stopped containers\n  stats       Display a live stream of container(s) resource usage statistics\n  stop        Stop one or more running containers\n  tag         Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE\n  top         Display the running processes of a container\n  unpause     Unpause all processes within one or more containers\n  update      Update configuration of one or more containers\n  version     Show the Docker version information\n  wait        Block until one or more containers stop, then print their exit codes\n\nRun \'docker COMMAND --help\' for more information on a command.\n'
    mock = mocker.patch('subprocess.Popen')
    mock.return_value.stdout = BytesIO(b'')
    mock.return_value.stderr = BytesIO(helptext_new)
    return mock

def output(cmd):
    if False:
        i = 10
        return i + 15
    return "docker: '{}' is not a docker command.\nSee 'docker --help'.".format(cmd)

def test_match():
    if False:
        return 10
    assert match(Command('docker pes', output('pes')))

@pytest.mark.usefixtures('no_memoize')
@pytest.mark.parametrize('script, output', [('docker swarn', output('swarn')), ('docker imge', output('imge'))])
def test_match_management_cmd(script, output):
    if False:
        for i in range(10):
            print('nop')
    assert match(Command(script, output))

@pytest.mark.usefixtures('no_memoize')
@pytest.mark.parametrize('script, output', [('docker swarm int', _DOCKER_SWARM_OUTPUT), ('docker image la', _DOCKER_IMAGE_OUTPUT)])
def test_match_management_subcmd(script, output):
    if False:
        for i in range(10):
            print('nop')
    assert match(Command(script, output))

@pytest.mark.parametrize('script, output', [('docker ps', ''), ('cat pes', output('pes'))])
def test_not_match(script, output):
    if False:
        print('Hello World!')
    assert not match(Command(script, output))

@pytest.mark.usefixtures('no_memoize', 'docker_help')
@pytest.mark.parametrize('wrong, fixed', [('pes', ['ps', 'push', 'pause']), ('tags', ['tag', 'stats', 'images'])])
def test_get_new_command(wrong, fixed):
    if False:
        for i in range(10):
            print('nop')
    command = Command('docker {}'.format(wrong), output(wrong))
    assert get_new_command(command) == ['docker {}'.format(x) for x in fixed]

@pytest.mark.usefixtures('no_memoize', 'docker_help_new')
@pytest.mark.parametrize('wrong, fixed', [('swarn', ['swarm', 'start', 'search']), ('inage', ['image', 'images', 'rename'])])
def test_get_new_management_command(wrong, fixed):
    if False:
        i = 10
        return i + 15
    command = Command('docker {}'.format(wrong), output(wrong))
    assert get_new_command(command) == ['docker {}'.format(x) for x in fixed]

@pytest.mark.usefixtures('no_memoize', 'docker_help_new')
@pytest.mark.parametrize('wrong, fixed, output', [('swarm int', ['swarm init', 'swarm join', 'swarm join-token'], _DOCKER_SWARM_OUTPUT), ('image la', ['image load', 'image ls', 'image tag'], _DOCKER_IMAGE_OUTPUT)])
def test_get_new_management_command_subcommand(wrong, fixed, output):
    if False:
        for i in range(10):
            print('nop')
    command = Command('docker {}'.format(wrong), output)
    assert get_new_command(command) == ['docker {}'.format(x) for x in fixed]