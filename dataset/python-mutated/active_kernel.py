from typing import Dict
from jupyter_client import KernelClient, KernelManager
from jupyter_client.kernelspec import NoSuchKernel
from mage_ai.server.kernels import DEFAULT_KERNEL_NAME, KernelName, kernel_managers
from mage_ai.server.logger import Logger
logger = Logger().new_server_logger(__name__)

class ActiveKernel:

    def __init__(self):
        if False:
            print('Hello World!')
        self.kernel = kernel_managers[DEFAULT_KERNEL_NAME]
        self.kernel_client = self.kernel.client()
active_kernel = ActiveKernel()

def switch_active_kernel(kernel_name: KernelName, emr_config: Dict=None) -> None:
    if False:
        print('Hello World!')
    "\n    Switches the active kernel to the specified kernel name, handling its startup and\n    shutdown.\n\n    This function switches the active kernel to the specified kernel name, shutting down any\n    currently active kernel and starting the new kernel. It also updates the active kernel and\n    its client. If the specified kernel is PySpark, it configures the active EMR cluster through\n    the 'emr_cluster_manager'.\n\n    This method logs various information and handles exceptions for different scenarios.\n\n    Args:\n        kernel_name (KernelName): The name of the kernel to switch to.\n        emr_config (Dict, optional): Configuration settings for EMR (Elastic MapReduce).\n            Defaults to None.\n\n    Returns:\n        None: This function does not return anything.\n\n    Raises:\n        NoSuchKernel: If the specified kernel is not available.\n        Exception: If the specified kernel is PySpark and is not installed,\n            it provides instructions for installation.\n\n    Note:\n        Ensure the necessary dependencies and configurations are set up for the desired kernels.\n    "
    logger.info(f'Switch active kernel: {kernel_name}')
    if kernel_managers[kernel_name].is_alive():
        logger.info(f'Kernel {kernel_name} is already alive.')
        return
    for kernel in kernel_managers.values():
        if kernel.is_alive():
            logger.info(f'Shut down current kernel {kernel}.')
            kernel.request_shutdown()
    try:
        new_kernel = kernel_managers[kernel_name]
        new_kernel.start_kernel()
        active_kernel.kernel = new_kernel
        active_kernel.kernel_client = new_kernel.client()
        if kernel_name == KernelName.PYSPARK:
            from mage_ai.cluster_manager.aws.emr_cluster_manager import emr_cluster_manager
            emr_cluster_manager.set_active_cluster(auto_selection=True, emr_config=emr_config)
    except NoSuchKernel as e:
        if kernel_name == KernelName.PYSPARK:
            raise Exception('PySpark kernel is not installed. Please follow the instructions in https://docs.mage.ai/integrations/spark-pyspark to install it.') from e
        else:
            raise e

def get_active_kernel() -> KernelManager:
    if False:
        while True:
            i = 10
    return active_kernel.kernel

def get_active_kernel_name() -> str:
    if False:
        while True:
            i = 10
    return active_kernel.kernel.kernel_name

def get_active_kernel_client() -> KernelClient:
    if False:
        i = 10
        return i + 15
    return active_kernel.kernel_client

def interrupt_kernel() -> None:
    if False:
        print('Hello World!')
    active_kernel.kernel.interrupt_kernel()

def restart_kernel() -> None:
    if False:
        return 10
    active_kernel.kernel.restart_kernel()
    active_kernel.kernel_client = active_kernel.kernel.client()

def start_kernel() -> None:
    if False:
        while True:
            i = 10
    active_kernel.kernel.start_kernel()
    active_kernel.kernel_client = active_kernel.kernel.client()