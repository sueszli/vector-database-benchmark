import os
import imp
import platform
main_code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
daemon = imp.load_source('daemon', '%s/pyspark/daemon.py' % main_code_dir)
if 'COVERAGE_PROCESS_START' in os.environ:
    if 'pypy' not in platform.python_implementation().lower():
        worker = imp.load_source('worker', '%s/pyspark/worker.py' % main_code_dir)

        def _cov_wrapped(*args, **kwargs):
            if False:
                print('Hello World!')
            import coverage
            cov = coverage.coverage(config_file=os.environ['COVERAGE_PROCESS_START'])
            cov.start()
            try:
                worker.main(*args, **kwargs)
            finally:
                cov.stop()
                cov.save()
        daemon.worker_main = _cov_wrapped
else:
    raise RuntimeError('COVERAGE_PROCESS_START environment variable is not set, exiting.')
if __name__ == '__main__':
    daemon.manager()