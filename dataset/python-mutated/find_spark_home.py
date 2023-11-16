import os
import sys

def _find_spark_home():
    if False:
        return 10
    'Find the SPARK_HOME.'
    if 'SPARK_HOME' in os.environ:
        return os.environ['SPARK_HOME']

    def is_spark_home(path):
        if False:
            return 10
        'Takes a path and returns true if the provided path could be a reasonable SPARK_HOME'
        return os.path.isfile(os.path.join(path, 'bin/spark-submit')) and (os.path.isdir(os.path.join(path, 'jars')) or os.path.isdir(os.path.join(path, 'assembly')))
    spark_dist_dir = 'spark-distribution'
    paths = ['../']
    if '__file__' in globals():
        paths += [os.path.join(os.path.dirname(os.path.realpath(__file__)), spark_dist_dir), os.path.dirname(os.path.realpath(__file__))]
    import_error_raised = False
    from importlib.util import find_spec
    try:
        module_home = os.path.dirname(find_spec('pyspark').origin)
        paths.append(os.path.join(module_home, spark_dist_dir))
        paths.append(module_home)
        paths.append(os.path.join(module_home, '../../'))
    except ImportError:
        import_error_raised = True
    paths = [os.path.abspath(p) for p in paths]
    try:
        return next((path for path in paths if is_spark_home(path)))
    except StopIteration:
        print('Could not find valid SPARK_HOME while searching {0}'.format(paths), file=sys.stderr)
        if import_error_raised:
            print("\nDid you install PySpark via a package manager such as pip or Conda? If so,\nPySpark was not found in your Python environment. It is possible your\nPython environment does not properly bind with your package manager.\n\nPlease check your default 'python' and if you set PYSPARK_PYTHON and/or\nPYSPARK_DRIVER_PYTHON environment variables, and see if you can import\nPySpark, for example, 'python -c 'import pyspark'.\n\nIf you cannot import, you can install by using the Python executable directly,\nfor example, 'python -m pip install pyspark [--user]'. Otherwise, you can also\nexplicitly set the Python executable, that has PySpark installed, to\nPYSPARK_PYTHON or PYSPARK_DRIVER_PYTHON environment variables, for example,\n'PYSPARK_PYTHON=python3 pyspark'.\n", file=sys.stderr)
        sys.exit(-1)
if __name__ == '__main__':
    print(_find_spark_home())