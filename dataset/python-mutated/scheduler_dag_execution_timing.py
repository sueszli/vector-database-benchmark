from __future__ import annotations
import gc
import os
import statistics
import sys
import textwrap
import time
from argparse import Namespace
from operator import attrgetter
import rich_click as click
from airflow.jobs.job import run_job
MAX_DAG_RUNS_ALLOWED = 1

class ShortCircuitExecutorMixin:
    """
    Mixin class to manage the scheduler state during the performance test run.
    """

    def __init__(self, dag_ids_to_watch, num_runs):
        if False:
            print('Hello World!')
        super().__init__()
        self.num_runs_per_dag = num_runs
        self.reset(dag_ids_to_watch)

    def reset(self, dag_ids_to_watch):
        if False:
            print('Hello World!')
        '\n        Capture the value that will determine when the scheduler is reset.\n        '
        self.dags_to_watch = {dag_id: Namespace(waiting_for=self.num_runs_per_dag, runs={}) for dag_id in dag_ids_to_watch}

    def change_state(self, key, state, info=None):
        if False:
            i = 10
            return i + 15
        '\n        Change the state of scheduler by waiting till the tasks is complete\n        and then shut down the scheduler after the task is complete\n        '
        from airflow.utils.state import TaskInstanceState
        super().change_state(key, state, info=info)
        (dag_id, _, execution_date, __) = key
        if dag_id not in self.dags_to_watch:
            return
        run = self.dags_to_watch[dag_id].runs.get(execution_date)
        if not run:
            import airflow.models
            run = list(airflow.models.DagRun.find(dag_id=dag_id, execution_date=execution_date))[0]
            self.dags_to_watch[dag_id].runs[execution_date] = run
        if run and all((t.state == TaskInstanceState.SUCCESS for t in run.get_task_instances())):
            self.dags_to_watch[dag_id].runs.pop(execution_date)
            self.dags_to_watch[dag_id].waiting_for -= 1
            if self.dags_to_watch[dag_id].waiting_for == 0:
                self.dags_to_watch.pop(dag_id)
            if not self.dags_to_watch:
                self.log.warning('STOPPING SCHEDULER -- all runs complete')
                self.job_runner.processor_agent._done = True
                return
        self.log.warning('WAITING ON %d RUNS', sum(map(attrgetter('waiting_for'), self.dags_to_watch.values())))

def get_executor_under_test(dotted_path):
    if False:
        i = 10
        return i + 15
    '\n    Create and return a MockExecutor\n    '
    from airflow.executors.executor_loader import ExecutorLoader
    if dotted_path == 'MockExecutor':
        from tests.test_utils.mock_executor import MockExecutor as executor
    else:
        executor = ExecutorLoader.load_executor(dotted_path)
        executor_cls = type(executor)

    class ShortCircuitExecutor(ShortCircuitExecutorMixin, executor_cls):
        """
        Placeholder class that implements the inheritance hierarchy
        """
        job_runner = None
    return ShortCircuitExecutor

def reset_dag(dag, session):
    if False:
        for i in range(10):
            print('nop')
    '\n    Delete all dag and task instances and then un_pause the Dag.\n    '
    import airflow.models
    DR = airflow.models.DagRun
    DM = airflow.models.DagModel
    TI = airflow.models.TaskInstance
    TF = airflow.models.TaskFail
    dag_id = dag.dag_id
    session.query(DM).filter(DM.dag_id == dag_id).update({'is_paused': False})
    session.query(DR).filter(DR.dag_id == dag_id).delete()
    session.query(TI).filter(TI.dag_id == dag_id).delete()
    session.query(TF).filter(TF.dag_id == dag_id).delete()

def pause_all_dags(session):
    if False:
        return 10
    '\n    Pause all Dags\n    '
    from airflow.models.dag import DagModel
    session.query(DagModel).update({'is_paused': True})

def create_dag_runs(dag, num_runs, session):
    if False:
        print('Hello World!')
    '\n    Create  `num_runs` of dag runs for sub-sequent schedules\n    '
    from airflow.utils import timezone
    from airflow.utils.state import DagRunState
    try:
        from airflow.utils.types import DagRunType
        id_prefix = f'{DagRunType.SCHEDULED.value}__'
    except ImportError:
        from airflow.models.dagrun import DagRun
        id_prefix = DagRun.ID_PREFIX
    last_dagrun_data_interval = None
    for _ in range(num_runs):
        next_info = dag.next_dagrun_info(last_dagrun_data_interval)
        logical_date = next_info.logical_date
        dag.create_dagrun(run_id=f'{id_prefix}{logical_date.isoformat()}', execution_date=logical_date, start_date=timezone.utcnow(), state=DagRunState.RUNNING, external_trigger=False, session=session)
        last_dagrun_data_interval = next_info.data_interval

@click.command()
@click.option('--num-runs', default=1, help='number of DagRun, to run for each DAG')
@click.option('--repeat', default=3, help='number of times to run test, to reduce variance')
@click.option('--pre-create-dag-runs', is_flag=True, default=False, help='Pre-create the dag runs and stop the scheduler creating more.\n\n        Warning: this makes the scheduler do (slightly) less work so may skew your numbers. Use sparingly!\n        ')
@click.option('--executor-class', default='MockExecutor', help=textwrap.dedent("\n          Dotted path Executor class to test, for example\n          'airflow.executors.local_executor.LocalExecutor'. Defaults to MockExecutor which doesn't run tasks.\n      "))
@click.argument('dag_ids', required=True, nargs=-1)
def main(num_runs, repeat, pre_create_dag_runs, executor_class, dag_ids):
    if False:
        while True:
            i = 10
    '\n    This script can be used to measure the total "scheduler overhead" of Airflow.\n\n    By overhead we mean if the tasks executed instantly as soon as they are\n    executed (i.e. they do nothing) how quickly could we schedule them.\n\n    It will monitor the task completion of the Mock/stub executor (no actual\n    tasks are run) and after the required number of dag runs for all the\n    specified dags have completed all their tasks, it will cleanly shut down\n    the scheduler.\n\n    The dags you run with need to have an early enough start_date to create the\n    desired number of runs.\n\n    Care should be taken that other limits (DAG max_active_tasks, pool size etc) are\n    not the bottleneck. This script doesn\'t help you in that regard.\n\n    It is recommended to repeat the test at least 3 times (`--repeat=3`, the\n    default) so that you can get somewhat-accurate variance on the reported\n    timing numbers, but this can be disabled for longer runs if needed.\n    '
    os.environ['AIRFLOW__CORE__UNIT_TEST_MODE'] = 'True'
    os.environ['AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG'] = '500'
    os.environ['AIRFLOW_BENCHMARK_MAX_DAG_RUNS'] = str(num_runs)
    os.environ['PERF_MAX_RUNS'] = str(num_runs)
    if pre_create_dag_runs:
        os.environ['AIRFLOW__SCHEDULER__USE_JOB_SCHEDULE'] = 'False'
    from airflow.jobs.job import Job
    from airflow.jobs.scheduler_job_runner import SchedulerJobRunner
    from airflow.models.dagbag import DagBag
    from airflow.utils import db
    dagbag = DagBag()
    dags = []
    with db.create_session() as session:
        pause_all_dags(session)
        for dag_id in dag_ids:
            dag = dagbag.get_dag(dag_id)
            dag.sync_to_db(session=session)
            dags.append(dag)
            reset_dag(dag, session)
            next_info = dag.next_dagrun_info(None)
            for _ in range(num_runs - 1):
                next_info = dag.next_dagrun_info(next_info.data_interval)
            end_date = dag.end_date or dag.default_args.get('end_date')
            if end_date != next_info.logical_date:
                message = f'DAG {dag_id} has incorrect end_date ({end_date}) for number of runs! It should be {next_info.logical_date}'
                sys.exit(message)
            if pre_create_dag_runs:
                create_dag_runs(dag, num_runs, session)
    ShortCircuitExecutor = get_executor_under_test(executor_class)
    executor = ShortCircuitExecutor(dag_ids_to_watch=dag_ids, num_runs=num_runs)
    scheduler_job = Job(executor=executor)
    job_runner = SchedulerJobRunner(job=scheduler_job, dag_ids=dag_ids, do_pickle=False)
    executor.job_runner = job_runner
    total_tasks = sum((len(dag.tasks) for dag in dags))
    if 'PYSPY' in os.environ:
        pid = str(os.getpid())
        filename = os.environ.get('PYSPY_O', 'flame-' + pid + '.html')
        os.spawnlp(os.P_NOWAIT, 'sudo', 'sudo', 'py-spy', 'record', '-o', filename, '-p', pid, '--idle')
    times = []
    code_to_test = lambda : run_job(job=job_runner.job, execute_callable=job_runner._execute)
    for count in range(repeat):
        if not count:
            with db.create_session() as session:
                for dag in dags:
                    reset_dag(dag, session)
            executor.reset(dag_ids)
            scheduler_job = Job(executor=executor)
            job_runner = SchedulerJobRunner(job=scheduler_job, dag_ids=dag_ids, do_pickle=False)
            executor.scheduler_job = scheduler_job
        gc.disable()
        start = time.perf_counter()
        code_to_test()
        times.append(time.perf_counter() - start)
        gc.enable()
        print(f'Run {count + 1} time: {times[-1]:.5f}')
    print()
    print()
    print(f'Time for {num_runs} dag runs of {len(dags)} dags with {total_tasks} total tasks: ', end='')
    if len(times) > 1:
        print(f'{statistics.mean(times):.4f}s (±{statistics.stdev(times):.3f}s)')
    else:
        print(f'{times[0]:.4f}s')
    print()
    print()
if __name__ == '__main__':
    main()