import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, NoReturn, Optional, Set, SupportsIndex
import yaml
from jinja2 import Environment, FileSystemLoader
MAIN_PROCESS_HTTP_LISTENER_PORT = 8080
MAIN_PROCESS_INSTANCE_NAME = 'main'
MAIN_PROCESS_LOCALHOST_ADDRESS = '127.0.0.1'
MAIN_PROCESS_REPLICATION_PORT = 9093
MAIN_PROCESS_UNIX_SOCKET_PUBLIC_PATH = '/run/main_public.sock'
MAIN_PROCESS_UNIX_SOCKET_PRIVATE_PATH = '/run/main_private.sock'
WORKER_PLACEHOLDER_NAME = 'placeholder_name'
WORKERS_CONFIG: Dict[str, Dict[str, Any]] = {'pusher': {'app': 'synapse.app.generic_worker', 'listener_resources': [], 'endpoint_patterns': [], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'user_dir': {'app': 'synapse.app.generic_worker', 'listener_resources': ['client'], 'endpoint_patterns': ['^/_matrix/client/(api/v1|r0|v3|unstable)/user_directory/search$'], 'shared_extra_conf': {'update_user_directory_from_worker': WORKER_PLACEHOLDER_NAME}, 'worker_extra_conf': ''}, 'media_repository': {'app': 'synapse.app.generic_worker', 'listener_resources': ['media'], 'endpoint_patterns': ['^/_matrix/media/', '^/_synapse/admin/v1/purge_media_cache$', '^/_synapse/admin/v1/room/.*/media.*$', '^/_synapse/admin/v1/user/.*/media.*$', '^/_synapse/admin/v1/media/.*$', '^/_synapse/admin/v1/quarantine_media/.*$'], 'shared_extra_conf': {'enable_media_repo': False, 'media_instance_running_background_jobs': WORKER_PLACEHOLDER_NAME}, 'worker_extra_conf': 'enable_media_repo: true'}, 'appservice': {'app': 'synapse.app.generic_worker', 'listener_resources': [], 'endpoint_patterns': [], 'shared_extra_conf': {'notify_appservices_from_worker': WORKER_PLACEHOLDER_NAME}, 'worker_extra_conf': ''}, 'federation_sender': {'app': 'synapse.app.generic_worker', 'listener_resources': [], 'endpoint_patterns': [], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'synchrotron': {'app': 'synapse.app.generic_worker', 'listener_resources': ['client'], 'endpoint_patterns': ['^/_matrix/client/(v2_alpha|r0|v3)/sync$', '^/_matrix/client/(api/v1|v2_alpha|r0|v3)/events$', '^/_matrix/client/(api/v1|r0|v3)/initialSync$', '^/_matrix/client/(api/v1|r0|v3)/rooms/[^/]+/initialSync$'], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'client_reader': {'app': 'synapse.app.generic_worker', 'listener_resources': ['client'], 'endpoint_patterns': ['^/_matrix/client/(api/v1|r0|v3|unstable)/publicRooms$', '^/_matrix/client/(api/v1|r0|v3|unstable)/rooms/.*/joined_members$', '^/_matrix/client/(api/v1|r0|v3|unstable)/rooms/.*/context/.*$', '^/_matrix/client/(api/v1|r0|v3|unstable)/rooms/.*/members$', '^/_matrix/client/(api/v1|r0|v3|unstable)/rooms/.*/state$', '^/_matrix/client/v1/rooms/.*/hierarchy$', '^/_matrix/client/(v1|unstable)/rooms/.*/relations/', '^/_matrix/client/v1/rooms/.*/threads$', '^/_matrix/client/(api/v1|r0|v3|unstable)/login$', '^/_matrix/client/(api/v1|r0|v3|unstable)/account/3pid$', '^/_matrix/client/(api/v1|r0|v3|unstable)/account/whoami$', '^/_matrix/client/versions$', '^/_matrix/client/(api/v1|r0|v3|unstable)/voip/turnServer$', '^/_matrix/client/(r0|v3|unstable)/register$', '^/_matrix/client/(r0|v3|unstable)/register/available$', '^/_matrix/client/(r0|v3|unstable)/auth/.*/fallback/web$', '^/_matrix/client/(api/v1|r0|v3|unstable)/rooms/.*/messages$', '^/_matrix/client/(api/v1|r0|v3|unstable)/rooms/.*/event', '^/_matrix/client/(api/v1|r0|v3|unstable)/joined_rooms', '^/_matrix/client/(api/v1|r0|v3|unstable/.*)/rooms/.*/aliases', '^/_matrix/client/v1/rooms/.*/timestamp_to_event$', '^/_matrix/client/(api/v1|r0|v3|unstable)/search', '^/_matrix/client/(r0|v3|unstable)/user/.*/filter(/|$)', '^/_matrix/client/(r0|v3|unstable)/password_policy$', '^/_matrix/client/(api/v1|r0|v3|unstable)/directory/room/.*$', '^/_matrix/client/(r0|v3|unstable)/capabilities$', '^/_matrix/client/(r0|v3|unstable)/notifications$'], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'federation_reader': {'app': 'synapse.app.generic_worker', 'listener_resources': ['federation'], 'endpoint_patterns': ['^/_matrix/federation/(v1|v2)/event/', '^/_matrix/federation/(v1|v2)/state/', '^/_matrix/federation/(v1|v2)/state_ids/', '^/_matrix/federation/(v1|v2)/backfill/', '^/_matrix/federation/(v1|v2)/get_missing_events/', '^/_matrix/federation/(v1|v2)/publicRooms', '^/_matrix/federation/(v1|v2)/query/', '^/_matrix/federation/(v1|v2)/make_join/', '^/_matrix/federation/(v1|v2)/make_leave/', '^/_matrix/federation/(v1|v2)/send_join/', '^/_matrix/federation/(v1|v2)/send_leave/', '^/_matrix/federation/(v1|v2)/invite/', '^/_matrix/federation/(v1|v2)/query_auth/', '^/_matrix/federation/(v1|v2)/event_auth/', '^/_matrix/federation/v1/timestamp_to_event/', '^/_matrix/federation/(v1|v2)/exchange_third_party_invite/', '^/_matrix/federation/(v1|v2)/user/devices/', '^/_matrix/federation/(v1|v2)/get_groups_publicised$', '^/_matrix/key/v2/query'], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'federation_inbound': {'app': 'synapse.app.generic_worker', 'listener_resources': ['federation'], 'endpoint_patterns': ['/_matrix/federation/(v1|v2)/send/'], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'event_persister': {'app': 'synapse.app.generic_worker', 'listener_resources': ['replication'], 'endpoint_patterns': [], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'background_worker': {'app': 'synapse.app.generic_worker', 'listener_resources': [], 'endpoint_patterns': [], 'shared_extra_conf': {'run_background_tasks_on': WORKER_PLACEHOLDER_NAME}, 'worker_extra_conf': ''}, 'event_creator': {'app': 'synapse.app.generic_worker', 'listener_resources': ['client'], 'endpoint_patterns': ['^/_matrix/client/(api/v1|r0|v3|unstable)/rooms/.*/redact', '^/_matrix/client/(api/v1|r0|v3|unstable)/rooms/.*/send', '^/_matrix/client/(api/v1|r0|v3|unstable)/rooms/.*/(join|invite|leave|ban|unban|kick)$', '^/_matrix/client/(api/v1|r0|v3|unstable)/join/', '^/_matrix/client/(api/v1|r0|v3|unstable)/knock/', '^/_matrix/client/(api/v1|r0|v3|unstable)/profile/'], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'frontend_proxy': {'app': 'synapse.app.generic_worker', 'listener_resources': ['client', 'replication'], 'endpoint_patterns': ['^/_matrix/client/(api/v1|r0|v3|unstable)/keys/upload'], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'account_data': {'app': 'synapse.app.generic_worker', 'listener_resources': ['client', 'replication'], 'endpoint_patterns': ['^/_matrix/client/(r0|v3|unstable)/.*/tags', '^/_matrix/client/(r0|v3|unstable)/.*/account_data'], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'presence': {'app': 'synapse.app.generic_worker', 'listener_resources': ['client', 'replication'], 'endpoint_patterns': ['^/_matrix/client/(api/v1|r0|v3|unstable)/presence/'], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'receipts': {'app': 'synapse.app.generic_worker', 'listener_resources': ['client', 'replication'], 'endpoint_patterns': ['^/_matrix/client/(r0|v3|unstable)/rooms/.*/receipt', '^/_matrix/client/(r0|v3|unstable)/rooms/.*/read_markers'], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'to_device': {'app': 'synapse.app.generic_worker', 'listener_resources': ['client', 'replication'], 'endpoint_patterns': ['^/_matrix/client/(r0|v3|unstable)/sendToDevice/'], 'shared_extra_conf': {}, 'worker_extra_conf': ''}, 'typing': {'app': 'synapse.app.generic_worker', 'listener_resources': ['client', 'replication'], 'endpoint_patterns': ['^/_matrix/client/(api/v1|r0|v3|unstable)/rooms/.*/typing'], 'shared_extra_conf': {}, 'worker_extra_conf': ''}}
NGINX_LOCATION_CONFIG_BLOCK = '\n    location ~* {endpoint} {{\n        proxy_pass {upstream};\n        proxy_set_header X-Forwarded-For $remote_addr;\n        proxy_set_header X-Forwarded-Proto $scheme;\n        proxy_set_header Host $host;\n    }}\n'
NGINX_UPSTREAM_CONFIG_BLOCK = '\nupstream {upstream_worker_base_name} {{\n{body}\n}}\n'

def log(txt: str) -> None:
    if False:
        i = 10
        return i + 15
    print(txt)

def error(txt: str) -> NoReturn:
    if False:
        for i in range(10):
            print('nop')
    print(txt, file=sys.stderr)
    sys.exit(2)

def flush_buffers() -> None:
    if False:
        print('Hello World!')
    sys.stdout.flush()
    sys.stderr.flush()

def convert(src: str, dst: str, **template_vars: object) -> None:
    if False:
        while True:
            i = 10
    'Generate a file from a template\n\n    Args:\n        src: Path to the input file.\n        dst: Path to write to.\n        template_vars: The arguments to replace placeholder variables in the template with.\n    '
    env = Environment(loader=FileSystemLoader(os.path.dirname(src)), autoescape=False)
    template = env.get_template(os.path.basename(src))
    rendered = template.render(**template_vars)
    with open(dst, 'a') as outfile:
        outfile.write('\n')
        outfile.write(rendered)

def add_worker_roles_to_shared_config(shared_config: dict, worker_types_set: Set[str], worker_name: str, worker_port: int) -> None:
    if False:
        print('Hello World!')
    'Given a dictionary representing a config file shared across all workers,\n    append appropriate worker information to it for the current worker_type instance.\n\n    Args:\n        shared_config: The config dict that all worker instances share (after being\n            converted to YAML)\n        worker_types_set: The type of worker (one of those defined in WORKERS_CONFIG).\n            This list can be a single worker type or multiple.\n        worker_name: The name of the worker instance.\n        worker_port: The HTTP replication port that the worker instance is listening on.\n    '
    instance_map = shared_config.setdefault('instance_map', {})
    singular_stream_writers = ['account_data', 'presence', 'receipts', 'to_device', 'typing']
    if 'pusher' in worker_types_set:
        shared_config.setdefault('pusher_instances', []).append(worker_name)
    if 'federation_sender' in worker_types_set:
        shared_config.setdefault('federation_sender_instances', []).append(worker_name)
    if 'event_persister' in worker_types_set:
        shared_config.setdefault('stream_writers', {}).setdefault('events', []).append(worker_name)
        if os.environ.get('SYNAPSE_USE_UNIX_SOCKET', False):
            instance_map[worker_name] = {'path': f'/run/worker.{worker_port}'}
        else:
            instance_map[worker_name] = {'host': 'localhost', 'port': worker_port}
    for worker in worker_types_set:
        if worker in singular_stream_writers:
            shared_config.setdefault('stream_writers', {}).setdefault(worker, []).append(worker_name)
            if os.environ.get('SYNAPSE_USE_UNIX_SOCKET', False):
                instance_map[worker_name] = {'path': f'/run/worker.{worker_port}'}
            else:
                instance_map[worker_name] = {'host': 'localhost', 'port': worker_port}

def merge_worker_template_configs(existing_dict: Optional[Dict[str, Any]], to_be_merged_dict: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    'When given an existing dict of worker template configuration consisting with both\n        dicts and lists, merge new template data from WORKERS_CONFIG(or create) and\n        return new dict.\n\n    Args:\n        existing_dict: Either an existing worker template or a fresh blank one.\n        to_be_merged_dict: The template from WORKERS_CONFIGS to be merged into\n            existing_dict.\n    Returns: The newly merged together dict values.\n    '
    new_dict: Dict[str, Any] = {}
    if not existing_dict:
        new_dict = to_be_merged_dict.copy()
    else:
        for i in to_be_merged_dict.keys():
            if i == 'endpoint_patterns' or i == 'listener_resources':
                new_dict[i] = list(set(existing_dict[i] + to_be_merged_dict[i]))
            elif i == 'shared_extra_conf':
                new_dict[i] = {**existing_dict[i], **to_be_merged_dict[i]}
            elif i == 'worker_extra_conf':
                new_dict[i] = existing_dict[i] + to_be_merged_dict[i]
            else:
                new_dict[i] = to_be_merged_dict[i]
    return new_dict

def insert_worker_name_for_worker_config(existing_dict: Dict[str, Any], worker_name: str) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    "Insert a given worker name into the worker's configuration dict.\n\n    Args:\n        existing_dict: The worker_config dict that is imported into shared_config.\n        worker_name: The name of the worker to insert.\n    Returns: Copy of the dict with newly inserted worker name\n    "
    dict_to_edit = existing_dict.copy()
    for (k, v) in dict_to_edit['shared_extra_conf'].items():
        if v == WORKER_PLACEHOLDER_NAME:
            dict_to_edit['shared_extra_conf'][k] = worker_name
    return dict_to_edit

def apply_requested_multiplier_for_worker(worker_types: List[str]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Apply multiplier(if found) by returning a new expanded list with some basic error\n    checking.\n\n    Args:\n        worker_types: The unprocessed List of requested workers\n    Returns:\n        A new list with all requested workers expanded.\n    '
    new_worker_types = []
    for worker_type in worker_types:
        if ':' in worker_type:
            worker_type_components = split_and_strip_string(worker_type, ':', 1)
            worker_count = 0
            try:
                worker_count = int(worker_type_components[1])
            except ValueError:
                error(f"Bad number in worker count for '{worker_type}': '{worker_type_components[1]}' is not an integer")
            for _ in range(worker_count):
                new_worker_types.append(worker_type_components[0])
        else:
            new_worker_types.append(worker_type)
    return new_worker_types

def is_sharding_allowed_for_worker_type(worker_type: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Helper to check to make sure worker types that cannot have multiples do not.\n\n    Args:\n        worker_type: The type of worker to check against.\n    Returns: True if allowed, False if not\n    '
    return worker_type not in ['background_worker', 'account_data', 'presence', 'receipts', 'typing', 'to_device']

def split_and_strip_string(given_string: str, split_char: str, max_split: SupportsIndex=-1) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper to split a string on split_char and strip whitespace from each end of each\n        element.\n    Args:\n        given_string: The string to split\n        split_char: The character to split the string on\n        max_split: kwarg for split() to limit how many times the split() happens\n    Returns:\n        A List of strings\n    '
    return [x.strip() for x in given_string.split(split_char, maxsplit=max_split)]

def generate_base_homeserver_config() -> None:
    if False:
        i = 10
        return i + 15
    'Starts Synapse and generates a basic homeserver config, which will later be\n    modified for worker support.\n\n    Raises: CalledProcessError if calling start.py returned a non-zero exit code.\n    '
    os.environ['SYNAPSE_HTTP_PORT'] = str(MAIN_PROCESS_HTTP_LISTENER_PORT)
    subprocess.run(['/usr/local/bin/python', '/start.py', 'migrate_config'], check=True)

def parse_worker_types(requested_worker_types: List[str]) -> Dict[str, Set[str]]:
    if False:
        return 10
    "Read the desired list of requested workers and prepare the data for use in\n        generating worker config files while also checking for potential gotchas.\n\n    Args:\n        requested_worker_types: The list formed from the split environment variable\n            containing the unprocessed requests for workers.\n\n    Returns: A dict of worker names to set of worker types. Format:\n        {'worker_name':\n            {'worker_type', 'worker_type2'}\n        }\n    "
    worker_base_name_counter: Dict[str, int] = defaultdict(int)
    worker_type_shard_counter: Dict[str, int] = defaultdict(int)
    dict_to_return: Dict[str, Set[str]] = {}
    multiple_processed_worker_types = apply_requested_multiplier_for_worker(requested_worker_types)
    for worker_type_string in multiple_processed_worker_types:
        worker_base_name: str = ''
        if '=' in worker_type_string:
            worker_type_split = split_and_strip_string(worker_type_string, '=')
            if len(worker_type_split) > 2:
                error(f"There should only be one '=' in the worker type string. Please fix: {worker_type_string}")
            worker_base_name = worker_type_split[0]
            if not re.match('^[a-zA-Z0-9_+-]*[a-zA-Z_+-]$', worker_base_name):
                error(f'Invalid worker name; please choose a name consisting of alphanumeric letters, _ + -, but not ending with a digit: {worker_base_name!r}')
            worker_type_string = worker_type_split[1]
        worker_types_set: Set[str] = set(split_and_strip_string(worker_type_string, '+'))
        if not worker_base_name:
            worker_base_name = '+'.join(sorted(worker_types_set))
        for worker_type in worker_types_set:
            if worker_type not in WORKERS_CONFIG:
                error(f"{worker_type} is an unknown worker type! Was found in '{worker_type_string}'. Please fix!")
            if worker_type in worker_type_shard_counter:
                if not is_sharding_allowed_for_worker_type(worker_type):
                    error(f'There can be only a single worker with {worker_type} type. Please recount and remove.')
            worker_type_shard_counter[worker_type] += 1
        worker_base_name_counter[worker_base_name] += 1
        worker_number = worker_base_name_counter[worker_base_name]
        worker_name = f'{worker_base_name}{worker_number}'
        if worker_number > 1:
            first_worker_with_base_name = dict_to_return[f'{worker_base_name}1']
            if first_worker_with_base_name != worker_types_set:
                error(f"Can not use worker_name: '{worker_name}' for worker_type(s): {worker_types_set!r}. It is already in use by worker_type(s): {first_worker_with_base_name!r}")
        dict_to_return[worker_name] = worker_types_set
    return dict_to_return

def generate_worker_files(environ: Mapping[str, str], config_path: str, data_dir: str, requested_worker_types: Dict[str, Set[str]]) -> None:
    if False:
        return 10
    "Read the desired workers(if any) that is passed in and generate shared\n        homeserver, nginx and supervisord configs.\n\n    Args:\n        environ: os.environ instance.\n        config_path: The location of the generated Synapse main worker config file.\n        data_dir: The location of the synapse data directory. Where log and\n            user-facing config files live.\n        requested_worker_types: A Dict containing requested workers in the format of\n            {'worker_name1': {'worker_type', ...}}\n    "
    using_unix_sockets = environ.get('SYNAPSE_USE_UNIX_SOCKET', False)
    listeners: List[Any]
    if using_unix_sockets:
        listeners = [{'path': MAIN_PROCESS_UNIX_SOCKET_PRIVATE_PATH, 'type': 'http', 'resources': [{'names': ['replication']}]}]
    else:
        listeners = [{'port': MAIN_PROCESS_REPLICATION_PORT, 'bind_address': MAIN_PROCESS_LOCALHOST_ADDRESS, 'type': 'http', 'resources': [{'names': ['replication']}]}]
    with open(config_path) as file_stream:
        original_config = yaml.safe_load(file_stream)
        original_listeners = original_config.get('listeners')
        if original_listeners:
            listeners += original_listeners
    shared_config: Dict[str, Any] = {'listeners': listeners}
    worker_descriptors: List[Dict[str, Any]] = []
    nginx_upstreams: Dict[str, Set[int]] = {}
    nginx_locations: Dict[str, str] = {}
    os.makedirs('/conf/workers', exist_ok=True)
    worker_port = 18009
    if using_unix_sockets:
        healthcheck_urls = [f'--unix-socket {MAIN_PROCESS_UNIX_SOCKET_PUBLIC_PATH} http://localhost/health']
    else:
        healthcheck_urls = ['http://localhost:8080/health']
    all_worker_types_in_use = set(chain(*requested_worker_types.values()))
    for worker_type in all_worker_types_in_use:
        for endpoint_pattern in WORKERS_CONFIG[worker_type]['endpoint_patterns']:
            nginx_locations[endpoint_pattern] = f'http://{worker_type}'
    for (worker_name, worker_types_set) in requested_worker_types.items():
        worker_config: Dict[str, Any] = {}
        for worker_type in worker_types_set:
            copy_of_template_config = WORKERS_CONFIG[worker_type].copy()
            worker_config = merge_worker_template_configs(worker_config, copy_of_template_config)
        worker_config = insert_worker_name_for_worker_config(worker_config, worker_name)
        worker_config.update({'name': worker_name, 'port': str(worker_port), 'config_path': config_path})
        worker_config['shared_extra_conf'].update(shared_config)
        shared_config = worker_config['shared_extra_conf']
        if using_unix_sockets:
            healthcheck_urls.append(f'--unix-socket /run/worker.{worker_port} http://localhost/health')
        else:
            healthcheck_urls.append('http://localhost:%d/health' % (worker_port,))
        add_worker_roles_to_shared_config(shared_config, worker_types_set, worker_name, worker_port)
        worker_descriptors.append(worker_config)
        log_config_filepath = generate_worker_log_config(environ, worker_name, data_dir)
        convert('/conf/worker.yaml.j2', f'/conf/workers/{worker_name}.yaml', **worker_config, worker_log_config_filepath=log_config_filepath, using_unix_sockets=using_unix_sockets)
        for worker_type in worker_types_set:
            nginx_upstreams.setdefault(worker_type, set()).add(worker_port)
        worker_port += 1
    nginx_location_config = ''
    for (endpoint, upstream) in nginx_locations.items():
        nginx_location_config += NGINX_LOCATION_CONFIG_BLOCK.format(endpoint=endpoint, upstream=upstream)
    nginx_upstream_config = ''
    for (upstream_worker_base_name, upstream_worker_ports) in nginx_upstreams.items():
        body = ''
        if using_unix_sockets:
            for port in upstream_worker_ports:
                body += f'    server unix:/run/worker.{port};\n'
        else:
            for port in upstream_worker_ports:
                body += f'    server localhost:{port};\n'
        nginx_upstream_config += NGINX_UPSTREAM_CONFIG_BLOCK.format(upstream_worker_base_name=upstream_worker_base_name, body=body)
    master_log_config = generate_worker_log_config(environ, 'master', data_dir)
    shared_config['log_config'] = master_log_config
    appservice_registrations = None
    appservice_registration_dir = os.environ.get('SYNAPSE_AS_REGISTRATION_DIR')
    if appservice_registration_dir:
        appservice_registrations = [str(reg_path.resolve()) for reg_path in Path(appservice_registration_dir).iterdir() if reg_path.suffix.lower() in ('.yaml', '.yml')]
    workers_in_use = len(requested_worker_types) > 0
    if workers_in_use:
        instance_map = shared_config.setdefault('instance_map', {})
        if using_unix_sockets:
            instance_map[MAIN_PROCESS_INSTANCE_NAME] = {'path': MAIN_PROCESS_UNIX_SOCKET_PRIVATE_PATH}
        else:
            instance_map[MAIN_PROCESS_INSTANCE_NAME] = {'host': MAIN_PROCESS_LOCALHOST_ADDRESS, 'port': MAIN_PROCESS_REPLICATION_PORT}
    convert('/conf/shared.yaml.j2', '/conf/workers/shared.yaml', shared_worker_config=yaml.dump(shared_config), appservice_registrations=appservice_registrations, enable_redis=workers_in_use, workers_in_use=workers_in_use, using_unix_sockets=using_unix_sockets)
    convert('/conf/nginx.conf.j2', '/etc/nginx/conf.d/matrix-synapse.conf', worker_locations=nginx_location_config, upstream_directives=nginx_upstream_config, tls_cert_path=os.environ.get('SYNAPSE_TLS_CERT'), tls_key_path=os.environ.get('SYNAPSE_TLS_KEY'), using_unix_sockets=using_unix_sockets)
    os.makedirs('/etc/supervisor', exist_ok=True)
    convert('/conf/supervisord.conf.j2', '/etc/supervisor/supervisord.conf', main_config_path=config_path, enable_redis=workers_in_use, using_unix_sockets=using_unix_sockets)
    convert('/conf/synapse.supervisord.conf.j2', '/etc/supervisor/conf.d/synapse.conf', workers=worker_descriptors, main_config_path=config_path, use_forking_launcher=environ.get('SYNAPSE_USE_EXPERIMENTAL_FORKING_LAUNCHER'))
    convert('/conf/healthcheck.sh.j2', '/healthcheck.sh', healthcheck_urls=healthcheck_urls)
    log_dir = data_dir + '/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

def generate_worker_log_config(environ: Mapping[str, str], worker_name: str, data_dir: str) -> str:
    if False:
        i = 10
        return i + 15
    'Generate a log.config file for the given worker.\n\n    Returns: the path to the generated file\n    '
    extra_log_template_args: Dict[str, Optional[str]] = {}
    if environ.get('SYNAPSE_WORKERS_WRITE_LOGS_TO_DISK'):
        extra_log_template_args['LOG_FILE_PATH'] = f'{data_dir}/logs/{worker_name}.log'
    extra_log_template_args['SYNAPSE_LOG_LEVEL'] = environ.get('SYNAPSE_LOG_LEVEL')
    extra_log_template_args['SYNAPSE_LOG_SENSITIVE'] = environ.get('SYNAPSE_LOG_SENSITIVE')
    extra_log_template_args['SYNAPSE_LOG_TESTING'] = environ.get('SYNAPSE_LOG_TESTING')
    log_config_filepath = f'/conf/workers/{worker_name}.log.config'
    convert('/conf/log.config', log_config_filepath, worker_name=worker_name, **extra_log_template_args, include_worker_name_in_log_line=environ.get('SYNAPSE_USE_EXPERIMENTAL_FORKING_LAUNCHER'))
    return log_config_filepath

def main(args: List[str], environ: MutableMapping[str, str]) -> None:
    if False:
        while True:
            i = 10
    config_dir = environ.get('SYNAPSE_CONFIG_DIR', '/data')
    config_path = environ.get('SYNAPSE_CONFIG_PATH', config_dir + '/homeserver.yaml')
    data_dir = environ.get('SYNAPSE_DATA_DIR', '/data')
    environ['SYNAPSE_NO_TLS'] = 'yes'
    if not os.path.exists(config_path):
        log('Generating base homeserver config')
        generate_base_homeserver_config()
    else:
        log('Base homeserver config exists—not regenerating')
    mark_filepath = '/conf/workers_have_been_configured'
    if not os.path.exists(mark_filepath):
        worker_types_env = environ.get('SYNAPSE_WORKER_TYPES', '').strip()
        if not worker_types_env:
            worker_types = []
            requested_worker_types: Dict[str, Any] = {}
        else:
            worker_types = split_and_strip_string(worker_types_env, ',')
            requested_worker_types = parse_worker_types(worker_types)
        log('Generating worker config files')
        generate_worker_files(environ, config_path, data_dir, requested_worker_types)
        with open(mark_filepath, 'w') as f:
            f.write('')
    else:
        log('Worker config exists—not regenerating')
    jemallocpath = '/usr/lib/%s-linux-gnu/libjemalloc.so.2' % (platform.machine(),)
    if os.path.isfile(jemallocpath):
        environ['LD_PRELOAD'] = jemallocpath
    else:
        log('Could not find %s, will not use' % (jemallocpath,))
    log('Starting supervisord')
    flush_buffers()
    os.execle('/usr/local/bin/supervisord', 'supervisord', '-c', '/etc/supervisor/supervisord.conf', environ)
if __name__ == '__main__':
    main(sys.argv, os.environ)