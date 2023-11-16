"""
This example shows how OnlineTool works when we need update prediction.
There are two parts including first_train and update_online_pred.
Firstly, we will finish the training and set the trained models to the `online` models.
Next, we will finish updating online predictions.
"""
import copy
import fire
import qlib
from qlib.constant import REG_CN
from qlib.model.trainer import task_train
from qlib.workflow.online.utils import OnlineToolR
from qlib.tests.config import CSI300_GBDT_TASK
task = copy.deepcopy(CSI300_GBDT_TASK)
task['record'] = {'class': 'SignalRecord', 'module_path': 'qlib.workflow.record_temp'}

class UpdatePredExample:

    def __init__(self, provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN, experiment_name='online_srv', task_config=task):
        if False:
            return 10
        qlib.init(provider_uri=provider_uri, region=region)
        self.experiment_name = experiment_name
        self.online_tool = OnlineToolR(self.experiment_name)
        self.task_config = task_config

    def first_train(self):
        if False:
            print('Hello World!')
        rec = task_train(self.task_config, experiment_name=self.experiment_name)
        self.online_tool.reset_online_tag(rec)

    def update_online_pred(self):
        if False:
            return 10
        self.online_tool.update_online_pred()

    def main(self):
        if False:
            i = 10
            return i + 15
        self.first_train()
        self.update_online_pred()
if __name__ == '__main__':
    fire.Fire(UpdatePredExample)