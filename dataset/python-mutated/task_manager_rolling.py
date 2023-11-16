"""
This example shows how a TrainerRM works based on TaskManager with rolling tasks.
After training, how to collect the rolling results will be shown in task_collecting.
Based on the ability of TaskManager, `worker` method offer a simple way for multiprocessing.
"""
from pprint import pprint
import fire
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager, run_task
from qlib.workflow.task.collect import RecorderCollector
from qlib.model.ens.group import RollingGroup
from qlib.model.trainer import TrainerR, TrainerRM, task_train
from qlib.tests.config import CSI100_RECORD_LGB_TASK_CONFIG, CSI100_RECORD_XGBOOST_TASK_CONFIG

class RollingTaskExample:

    def __init__(self, provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN, task_url='mongodb://10.0.0.4:27017/', task_db_name='rolling_db', experiment_name='rolling_exp', task_pool=None, task_config=None, rolling_step=550, rolling_type=RollingGen.ROLL_SD):
        if False:
            while True:
                i = 10
        if task_config is None:
            task_config = [CSI100_RECORD_XGBOOST_TASK_CONFIG, CSI100_RECORD_LGB_TASK_CONFIG]
        mongo_conf = {'task_url': task_url, 'task_db_name': task_db_name}
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.experiment_name = experiment_name
        if task_pool is None:
            self.trainer = TrainerR(experiment_name=self.experiment_name)
        else:
            self.task_pool = task_pool
            self.trainer = TrainerRM(self.experiment_name, self.task_pool)
        self.task_config = task_config
        self.rolling_gen = RollingGen(step=rolling_step, rtype=rolling_type)

    def reset(self):
        if False:
            return 10
        print('========== reset ==========')
        if isinstance(self.trainer, TrainerRM):
            TaskManager(task_pool=self.task_pool).remove()
        exp = R.get_exp(experiment_name=self.experiment_name)
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

    def task_generating(self):
        if False:
            for i in range(10):
                print('nop')
        print('========== task_generating ==========')
        tasks = task_generator(tasks=self.task_config, generators=self.rolling_gen)
        pprint(tasks)
        return tasks

    def task_training(self, tasks):
        if False:
            return 10
        print('========== task_training ==========')
        self.trainer.train(tasks)

    def worker(self):
        if False:
            for i in range(10):
                print('nop')
        print('========== worker ==========')
        run_task(task_train, self.task_pool, experiment_name=self.experiment_name)

    def task_collecting(self):
        if False:
            while True:
                i = 10
        print('========== task_collecting ==========')

        def rec_key(recorder):
            if False:
                for i in range(10):
                    print('nop')
            task_config = recorder.load_object('task')
            model_key = task_config['model']['class']
            rolling_key = task_config['dataset']['kwargs']['segments']['test']
            return (model_key, rolling_key)

        def my_filter(recorder):
            if False:
                while True:
                    i = 10
            (model_key, rolling_key) = rec_key(recorder)
            if model_key == 'LGBModel':
                return True
            return False
        collector = RecorderCollector(experiment=self.experiment_name, process_list=RollingGroup(), rec_key_func=rec_key, rec_filter_func=my_filter)
        print(collector())

    def main(self):
        if False:
            i = 10
            return i + 15
        self.reset()
        tasks = self.task_generating()
        self.task_training(tasks)
        self.task_collecting()
if __name__ == '__main__':
    fire.Fire(RollingTaskExample)