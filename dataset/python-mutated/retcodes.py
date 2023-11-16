"""
Module containing the logic for exit codes for the luigi binary. It's useful
when you in a programmatic way need to know if luigi actually finished the
given task, and if not why.
"""
import luigi
import sys
import logging
from luigi import IntParameter
from luigi.setup_logging import InterfaceLogging

class retcode(luigi.Config):
    """
    See the :ref:`return codes configuration section <retcode-config>`.
    """
    unhandled_exception = IntParameter(default=4, description='For internal luigi errors.')
    missing_data = IntParameter(default=0, description='For when there are incomplete ExternalTask dependencies.')
    task_failed = IntParameter(default=0, description="For when a task's run() method fails.")
    already_running = IntParameter(default=0, description='For both local --lock and luigid "lock"')
    scheduling_error = IntParameter(default=0, description="For when a task's complete() or requires() fails,\n                                                   or task-limit reached")
    not_run = IntParameter(default=0, description='For when a task is not granted run permission by the scheduler.')

def run_with_retcodes(argv):
    if False:
        i = 10
        return i + 15
    "\n    Run luigi with command line parsing, but raise ``SystemExit`` with the configured exit code.\n\n    Note: Usually you use the luigi binary directly and don't call this function yourself.\n\n    :param argv: Should (conceptually) be ``sys.argv[1:]``\n    "
    logger = logging.getLogger('luigi-interface')
    with luigi.cmdline_parser.CmdlineParser.global_instance(argv):
        retcodes = retcode()
    worker = None
    try:
        worker = luigi.interface._run(argv).worker
    except luigi.interface.PidLockAlreadyTakenExit:
        sys.exit(retcodes.already_running)
    except Exception:
        env_params = luigi.interface.core()
        InterfaceLogging.setup(env_params)
        logger.exception('Uncaught exception in luigi')
        sys.exit(retcodes.unhandled_exception)
    with luigi.cmdline_parser.CmdlineParser.global_instance(argv):
        task_sets = luigi.execution_summary._summary_dict(worker)
        root_task = luigi.execution_summary._root_task(worker)
        non_empty_categories = {k: v for (k, v) in task_sets.items() if v}.keys()

    def has(status):
        if False:
            while True:
                i = 10
        assert status in luigi.execution_summary._ORDERED_STATUSES
        return status in non_empty_categories
    codes_and_conds = ((retcodes.missing_data, has('still_pending_ext')), (retcodes.task_failed, has('failed')), (retcodes.already_running, has('run_by_other_worker')), (retcodes.scheduling_error, has('scheduling_error')), (retcodes.not_run, has('not_run')))
    expected_ret_code = max((code * (1 if cond else 0) for (code, cond) in codes_and_conds))
    if expected_ret_code == 0 and root_task not in task_sets['completed'] and (root_task not in task_sets['already_done']):
        sys.exit(retcodes.not_run)
    else:
        sys.exit(expected_ret_code)