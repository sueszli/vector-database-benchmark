import subprocess
from .common import is_windows, DEVNULL

def exec_cmd(cmd, nice=20, wait=True):
    if False:
        print('Hello World!')
    " Execute a child process from command in a new process\n    :param list|str cmd: sequence of program arguments or a single string. On Unix single string is interpreted\n    as the path of the program to execute, but it's only working if not passing arguments to the program.\n    :param int nice: *Default: 20 * process priority to bet set (Unix only). For windows lowest priority is always set.\n    :param bool wait: *Default: True* if True, program will wait for child process to terminate\n    :return:\n    "
    if is_windows():
        pc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=DEVNULL)
        (stdout, stderr) = pc.communicate()
        import win32process
        import win32api
        import win32con
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pc.pid)
        win32process.SetPriorityClass(handle, win32process.IDLE_PRIORITY_CLASS)
    else:
        pc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = pc.communicate()
    if wait:
        pc.wait()
    print(str(stderr) + '\n' + str(stdout))