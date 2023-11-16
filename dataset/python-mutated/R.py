import os
import sys
from importlib import util as imp_util, machinery as imp_machinery
from tempfile import NamedTemporaryFile
from .util import to_bytes
R_FUNCTIONS = {}
R_PACKAGE_PATHS = None
RDS_FILE_PATH = None
R_CONTAINER_IMAGE = None
METAFLOW_R_VERSION = None
R_VERSION = None
R_VERSION_CODE = None

def call_r(func_name, args):
    if False:
        i = 10
        return i + 15
    R_FUNCTIONS[func_name](*args)

def get_r_func(func_name):
    if False:
        print('Hello World!')
    return R_FUNCTIONS[func_name]

def package_paths():
    if False:
        return 10
    if R_PACKAGE_PATHS is not None:
        root = R_PACKAGE_PATHS['package']
        prefixlen = len('%s/' % root.rstrip('/'))
        for (path, dirs, files) in os.walk(R_PACKAGE_PATHS['package']):
            if '/.' in path:
                continue
            for fname in files:
                if fname[0] == '.':
                    continue
                p = os.path.join(path, fname)
                yield (p, os.path.join('metaflow-r', p[prefixlen:]))
        flow = R_PACKAGE_PATHS['flow']
        yield (flow, os.path.basename(flow))

def entrypoint():
    if False:
        return 10
    return 'PYTHONPATH=/root/metaflow R_LIBS_SITE=`Rscript -e \'cat(paste(.libPaths(), collapse=\\":\\"))\'`:metaflow/ Rscript metaflow-r/run_batch.R --flowRDS=%s' % RDS_FILE_PATH

def use_r():
    if False:
        return 10
    return R_PACKAGE_PATHS is not None

def container_image():
    if False:
        for i in range(10):
            print('nop')
    return R_CONTAINER_IMAGE

def metaflow_r_version():
    if False:
        return 10
    return METAFLOW_R_VERSION

def r_version():
    if False:
        while True:
            i = 10
    return R_VERSION

def r_version_code():
    if False:
        for i in range(10):
            print('nop')
    return R_VERSION_CODE

def working_dir():
    if False:
        while True:
            i = 10
    if use_r():
        return R_PACKAGE_PATHS['wd']
    return None

def load_module_from_path(module_name: str, path: str):
    if False:
        print('Hello World!')
    '\n    Loads a module from a given path\n\n    Parameters\n    ----------\n    module_name: str\n        Name to assign for the loaded module. Usable for importing after loading.\n    path: str\n        path to the file to be loaded\n    '
    loader = imp_machinery.SourceFileLoader(module_name, path)
    spec = imp_util.spec_from_loader(loader.name, loader)
    module = imp_util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[module_name] = module
    return module

def run(flow_script, r_functions, rds_file, metaflow_args, full_cmdline, r_paths, r_container_image, metaflow_r_version, r_version, r_version_code):
    if False:
        for i in range(10):
            print('nop')
    global R_FUNCTIONS, R_PACKAGE_PATHS, RDS_FILE_PATH, R_CONTAINER_IMAGE, METAFLOW_R_VERSION, R_VERSION, R_VERSION_CODE
    R_FUNCTIONS = r_functions
    R_PACKAGE_PATHS = r_paths
    RDS_FILE_PATH = rds_file
    R_CONTAINER_IMAGE = r_container_image
    METAFLOW_R_VERSION = metaflow_r_version
    R_VERSION = r_version
    R_VERSION_CODE = r_version_code
    if not isinstance(metaflow_args, list):
        metaflow_args = [metaflow_args]
    full_cmdline[0] = os.path.basename(full_cmdline[0])
    with NamedTemporaryFile(prefix='metaflowR.', delete=False) as tmp:
        tmp.write(to_bytes(flow_script))
    module = load_module_from_path('metaflowR', tmp.name)
    flow = module.FLOW(use_cli=False)
    from . import exception
    from . import cli
    try:
        cli.main(flow, args=metaflow_args, handle_exceptions=False, entrypoint=full_cmdline[:-len(metaflow_args)])
    except exception.MetaflowException as e:
        cli.print_metaflow_exception(e)
        os.remove(tmp.name)
        os._exit(1)
    except Exception as e:
        import sys
        print(e)
        sys.stdout.flush()
        os.remove(tmp.name)
        os._exit(1)
    finally:
        os.remove(tmp.name)