import glob
import logging
import os
import sys

def _is_flink_home(path):
    if False:
        i = 10
        return i + 15
    flink_script_file = path + '/bin/flink'
    if len(glob.glob(flink_script_file)) > 0:
        return True
    else:
        return False

def _is_apache_flink_libraries_home(path):
    if False:
        i = 10
        return i + 15
    flink_dist_jar_file = path + '/lib/flink-dist*.jar'
    if len(glob.glob(flink_dist_jar_file)) > 0:
        return True
    else:
        return False

def _find_flink_home():
    if False:
        while True:
            i = 10
    '\n    Find the FLINK_HOME.\n    '
    if 'FLINK_HOME' in os.environ:
        return os.environ['FLINK_HOME']
    else:
        try:
            current_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
            flink_root_dir = os.path.abspath(current_dir + '/../../')
            build_target = glob.glob(flink_root_dir + '/flink-dist/target/flink-*-bin/flink-*')
            if len(build_target) > 0 and _is_flink_home(build_target[0]):
                os.environ['FLINK_HOME'] = build_target[0]
                return build_target[0]
            FLINK_HOME = None
            for module_home in __import__('pyflink').__path__:
                if _is_apache_flink_libraries_home(module_home):
                    os.environ['FLINK_LIB_DIR'] = os.path.join(module_home, 'lib')
                    os.environ['FLINK_PLUGINS_DIR'] = os.path.join(module_home, 'plugins')
                    os.environ['FLINK_OPT_DIR'] = os.path.join(module_home, 'opt')
                if _is_flink_home(module_home):
                    FLINK_HOME = module_home
            if FLINK_HOME is not None:
                os.environ['FLINK_HOME'] = FLINK_HOME
                return FLINK_HOME
        except Exception:
            pass
        logging.error('Could not find valid FLINK_HOME(Flink distribution directory) in current environment.')
        sys.exit(-1)

def _find_flink_source_root():
    if False:
        while True:
            i = 10
    '\n    Find the flink source root directory.\n    '
    try:
        return os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../')
    except Exception:
        pass
    logging.error('Could not find valid flink source root directory in current environment.')
    sys.exit(-1)
if __name__ == '__main__':
    print(_find_flink_home())