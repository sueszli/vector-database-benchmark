"""Utilities for multi-client setup."""
import os
import sys
from absl import flags
import portpicker
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.python.platform import test as tf_test
_NUM_LOCAL_DEVICES = flags.DEFINE_integer('num_local_devices', 4, 'Number of local devices. 4 is the only allowed value for TPU.')
_NUM_CLIENTS = flags.DEFINE_integer('num_clients', 2, 'Number of clients. 0 for local mode. 2 is the only allowed value for TPU.')

def multi_client_main(client_config_function):
    if False:
        while True:
            i = 10
    'Creates a Flock of TensorFlow Processes on localhost.'
    flags.FLAGS(sys.argv, known_only=True)
    num_clients = _NUM_CLIENTS.value
    num_process = num_clients or 1
    num_local_devices = _NUM_LOCAL_DEVICES.value
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HIP_VISIBLE_DEVICES'] = ''
    mp_context = test_backend_util.get_mp_context()
    print('Check per client log in Test artifacts.', flush=True)
    server_ports = sorted([portpicker.pick_unused_port() for _ in range(num_process)], reverse=True)
    additional_ports = sorted([portpicker.pick_unused_port() for _ in range(num_process)])
    procs = []
    for client_idx in range(num_process):
        proc = mp_context.Process(target=run_client, args=(client_idx, num_clients, server_ports, additional_ports, num_local_devices, client_config_function), name=f'Client-{client_idx}')
        proc.start()
        procs.append(proc)
    exitcode = 0
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            exitcode = proc.exitcode
    sys.exit(exitcode)

def run_client(idx, num_clients, server_ports, additional_ports, num_local_devices, client_config_function):
    if False:
        return 10
    "Runs test.main() from a DTensor Client process on localhost.\n\n  This function runs in a separate process so that the eager context is\n  properly separated, which resembles real world multi-client setup.\n\n  Virtual devices are configured before test.main() is called.\n\n  Each client is configured to only have access to the physical GPU device\n  corresponding to its client id via CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES.\n\n  Each client is configured to only have access to some TPU cores\n  corresponding to its client id via flags.\n\n  The clients redirect stdout and stderr to files under Test Artifacts.\n\n  Args:\n    idx: integer task number represents the client's id from global picture.\n    num_clients: total number of clients.\n    server_ports: A list of ports that is allocated and to be used to construct\n      GRPC server. server_ports[idx] will be the GRPC server on the\n      corresponding client.\n    additional_ports: A list of ports that is allocated and to be used to\n      construct the backends.\n    num_local_devices: Number of devices per client.\n    client_config_function: A function, for each of the client to config the\n      local environment variables, etc. Note that the function will be called\n      with a dict of extra params, eg:\n        {'num_clients': 2\n         'client_id': 0,\n         'worker_jobs': ['localhost:port1', 'localhost:port2'],\n         'num_devices': 4,\n        }\n  "
    test_backend_util.slice_host_devices_for_multiworker(num_clients, idx, additional_ports)
    artifact_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', '')
    if artifact_dir:
        with open(os.path.join(artifact_dir, f'test-client-process-{idx}.log'), 'wb') as fp:
            os.dup2(fp.fileno(), 1)
            os.dup2(fp.fileno(), 2)
    worker_jobs = [f'localhost:{port:06d}' for port in server_ports]
    client_config_func_param = {'num_clients': num_clients, 'client_id': idx, 'worker_jobs': worker_jobs, 'num_devices': num_local_devices}
    client_config_function(client_config_func_param)
    tf_test.main()