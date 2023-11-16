import sys
from ipykernel import kernelspec
default_make_ipkernel_cmd = kernelspec.make_ipkernel_cmd

def custom_make_ipkernel_cmd(*args, **kwargs):
    if False:
        return 10
    '\n    Build modified Popen command list for launching an IPython kernel with MPI.\n\n    Parameters\n    ----------\n    *args : iterable\n        Additional positional arguments to be passed in `default_make_ipkernel_cmd`.\n    **kwargs : dict\n        Additional keyword arguments to be passed in `default_make_ipkernel_cmd`.\n\n    Returns\n    -------\n    array\n        A Popen command list.\n\n    Notes\n    -----\n    The parameters of the function should be kept in sync with the ones of the original function.\n    '
    mpi_arguments = ['mpiexec', '-n', '1']
    arguments = default_make_ipkernel_cmd(*args, **kwargs)
    return mpi_arguments + arguments
kernelspec.make_ipkernel_cmd = custom_make_ipkernel_cmd
if __name__ == '__main__':
    kernel_name = 'python3mpi'
    display_name = 'Python 3 (ipykernel) with MPI'
    dest = kernelspec.install(kernel_name=kernel_name, display_name=display_name, prefix=sys.prefix)
    print(f'Installed kernelspec {kernel_name} in {dest}')