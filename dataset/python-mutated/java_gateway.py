import importlib
import os
import shlex
import shutil
import struct
import tempfile
import time
from logging import WARN
from threading import RLock
from py4j.java_gateway import java_import, logger, JavaGateway, GatewayParameters, CallbackServerParameters
from pyflink.find_flink_home import _find_flink_home
from pyflink.pyflink_gateway_server import launch_gateway_server_process
from pyflink.util.exceptions import install_exception_handler, install_py4j_hooks
_gateway = None
_lock = RLock()

def is_launch_gateway_disabled():
    if False:
        print('Hello World!')
    if 'PYFLINK_GATEWAY_DISABLED' in os.environ and os.environ['PYFLINK_GATEWAY_DISABLED'].lower() not in ['0', 'false', '']:
        return True
    else:
        return False

def get_gateway():
    if False:
        print('Hello World!')
    global _gateway
    global _lock
    with _lock:
        if _gateway is None:
            logger.level = WARN
            if 'PYFLINK_GATEWAY_PORT' in os.environ:
                gateway_port = int(os.environ['PYFLINK_GATEWAY_PORT'])
                gateway_param = GatewayParameters(port=gateway_port, auto_convert=True)
                _gateway = JavaGateway(gateway_parameters=gateway_param, callback_server_parameters=CallbackServerParameters(port=0, daemonize=True, daemonize_connections=True))
            else:
                _gateway = launch_gateway()
            callback_server = _gateway.get_callback_server()
            callback_server_listening_address = callback_server.get_listening_address()
            callback_server_listening_port = callback_server.get_listening_port()
            _gateway.jvm.org.apache.flink.client.python.PythonEnvUtils.resetCallbackClient(_gateway.java_gateway_server, callback_server_listening_address, callback_server_listening_port)
            import_flink_view(_gateway)
            install_exception_handler()
            install_py4j_hooks()
            _gateway.entry_point.put('PythonFunctionFactory', PythonFunctionFactory())
            _gateway.entry_point.put('Watchdog', Watchdog())
    return _gateway

def launch_gateway():
    if False:
        for i in range(10):
            print('nop')
    '\n    launch jvm gateway\n    '
    if is_launch_gateway_disabled():
        raise Exception("It's launching the PythonGatewayServer during Python UDF execution which is unexpected. It usually happens when the job codes are in the top level of the Python script file and are not enclosed in a `if name == 'main'` statement.")
    args = ['-c', 'org.apache.flink.client.python.PythonGatewayServer']
    submit_args = os.environ.get('SUBMIT_ARGS', 'local')
    args += shlex.split(submit_args)
    conn_info_dir = tempfile.mkdtemp()
    try:
        (fd, conn_info_file) = tempfile.mkstemp(dir=conn_info_dir)
        os.close(fd)
        os.unlink(conn_info_file)
        _find_flink_home()
        env = dict(os.environ)
        env['_PYFLINK_CONN_INFO_PATH'] = conn_info_file
        p = launch_gateway_server_process(env, args)
        while not p.poll() and (not os.path.isfile(conn_info_file)):
            time.sleep(0.1)
        if not os.path.isfile(conn_info_file):
            stderr_info = p.stderr.read().decode('utf-8')
            raise RuntimeError('Java gateway process exited before sending its port number.\nStderr:\n' + stderr_info)
        with open(conn_info_file, 'rb') as info:
            gateway_port = struct.unpack('!I', info.read(4))[0]
    finally:
        shutil.rmtree(conn_info_dir)
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=gateway_port, auto_convert=True), callback_server_parameters=CallbackServerParameters(port=0, daemonize=True, daemonize_connections=True))
    return gateway

def import_flink_view(gateway):
    if False:
        print('Hello World!')
    '\n    import the classes used by PyFlink.\n    :param gateway:gateway connected to JavaGateWayServer\n    '
    java_import(gateway.jvm, 'org.apache.flink.table.api.*')
    java_import(gateway.jvm, 'org.apache.flink.table.api.config.*')
    java_import(gateway.jvm, 'org.apache.flink.table.api.java.*')
    java_import(gateway.jvm, 'org.apache.flink.table.api.bridge.java.*')
    java_import(gateway.jvm, 'org.apache.flink.table.api.dataview.*')
    java_import(gateway.jvm, 'org.apache.flink.table.catalog.*')
    java_import(gateway.jvm, 'org.apache.flink.table.descriptors.*')
    java_import(gateway.jvm, 'org.apache.flink.table.descriptors.python.*')
    java_import(gateway.jvm, 'org.apache.flink.table.expressions.*')
    java_import(gateway.jvm, 'org.apache.flink.table.sources.*')
    java_import(gateway.jvm, 'org.apache.flink.table.sinks.*')
    java_import(gateway.jvm, 'org.apache.flink.table.sources.*')
    java_import(gateway.jvm, 'org.apache.flink.table.types.*')
    java_import(gateway.jvm, 'org.apache.flink.table.types.logical.*')
    java_import(gateway.jvm, 'org.apache.flink.table.util.python.*')
    java_import(gateway.jvm, 'org.apache.flink.api.common.python.*')
    java_import(gateway.jvm, 'org.apache.flink.api.common.typeinfo.TypeInformation')
    java_import(gateway.jvm, 'org.apache.flink.api.common.typeinfo.Types')
    java_import(gateway.jvm, 'org.apache.flink.api.java.ExecutionEnvironment')
    java_import(gateway.jvm, 'org.apache.flink.streaming.api.environment.StreamExecutionEnvironment')
    java_import(gateway.jvm, 'org.apache.flink.api.common.restartstrategy.RestartStrategies')
    java_import(gateway.jvm, 'org.apache.flink.python.util.PythonDependencyUtils')
    java_import(gateway.jvm, 'org.apache.flink.python.PythonOptions')
    java_import(gateway.jvm, 'org.apache.flink.client.python.PythonGatewayServer')
    java_import(gateway.jvm, 'org.apache.flink.streaming.api.functions.python.*')
    java_import(gateway.jvm, 'org.apache.flink.streaming.api.operators.python.process.*')
    java_import(gateway.jvm, 'org.apache.flink.streaming.api.operators.python.embedded.*')
    java_import(gateway.jvm, 'org.apache.flink.streaming.api.typeinfo.python.*')

class PythonFunctionFactory(object):
    """
    Used to create PythonFunction objects for Java jobs.
    """

    def getPythonFunction(self, moduleName, objectName):
        if False:
            i = 10
            return i + 15
        udf_wrapper = getattr(importlib.import_module(moduleName), objectName)
        return udf_wrapper._java_user_defined_function()

    class Java:
        implements = ['org.apache.flink.client.python.PythonFunctionFactory']

class Watchdog(object):
    """
    Used to provide to Java side to check whether its parent process is alive.
    """

    def ping(self):
        if False:
            print('Hello World!')
        time.sleep(10)
        return True

    class Java:
        implements = ['org.apache.flink.client.python.PythonGatewayServer$Watchdog']