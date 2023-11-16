import os
from mmpt.utils import recursive_config

class BaseJob(object):

    def __init__(self, yaml_file, dryrun=False):
        if False:
            print('Hello World!')
        self.yaml_file = yaml_file
        self.config = recursive_config(yaml_file)
        self.dryrun = dryrun

    def submit(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def _normalize_cmd(self, cmd_list):
        if False:
            while True:
                i = 10
        cmd_list = list(cmd_list)
        yaml_index = cmd_list.index('[yaml]')
        cmd_list[yaml_index] = self.yaml_file
        return cmd_list

class LocalJob(BaseJob):
    CMD_CONFIG = {'local_single': ['fairseq-train', '[yaml]', '--user-dir', 'mmpt', '--task', 'mmtask', '--arch', 'mmarch', '--criterion', 'mmloss'], 'local_small': ['fairseq-train', '[yaml]', '--user-dir', 'mmpt', '--task', 'mmtask', '--arch', 'mmarch', '--criterion', 'mmloss', '--distributed-world-size', '2'], 'local_big': ['fairseq-train', '[yaml]', '--user-dir', 'mmpt', '--task', 'mmtask', '--arch', 'mmarch', '--criterion', 'mmloss', '--distributed-world-size', '8'], 'local_predict': ['python', 'mmpt_cli/predict.py', '[yaml]']}

    def __init__(self, yaml_file, job_type=None, dryrun=False):
        if False:
            return 10
        super().__init__(yaml_file, dryrun)
        if job_type is None:
            self.job_type = 'local_single'
            if self.config.task_type is not None:
                self.job_type = self.config.task_type
        else:
            self.job_type = job_type
        if self.job_type in ['local_single', 'local_small']:
            if self.config.fairseq.dataset.batch_size > 32:
                print('decreasing batch_size to 32 for local testing?')

    def submit(self):
        if False:
            for i in range(10):
                print('nop')
        cmd_list = self._normalize_cmd(LocalJob.CMD_CONFIG[self.job_type])
        if 'predict' not in self.job_type:
            from mmpt.utils import load_config
            config = load_config(config_file=self.yaml_file)
            for field in config.fairseq:
                for key in config.fairseq[field]:
                    if key in ['fp16', 'reset_optimizer', 'reset_dataloader', 'reset_meters']:
                        param = ['--' + key.replace('_', '-')]
                    else:
                        if key == 'lr':
                            value = str(config.fairseq[field][key][0])
                        elif key == 'adam_betas':
                            value = "'" + str(config.fairseq[field][key]) + "'"
                        else:
                            value = str(config.fairseq[field][key])
                        param = ['--' + key.replace('_', '-'), value]
                    cmd_list.extend(param)
        print('launching', ' '.join(cmd_list))
        if not self.dryrun:
            os.system(' '.join(cmd_list))
        return JobStatus('12345678')

class JobStatus(object):

    def __init__(self, job_id):
        if False:
            print('Hello World!')
        self.job_id = job_id

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.job_id

    def __str__(self):
        if False:
            print('Hello World!')
        return self.job_id

    def done(self):
        if False:
            i = 10
            return i + 15
        return False

    def running(self):
        if False:
            print('Hello World!')
        return False

    def result(self):
        if False:
            for i in range(10):
                print('nop')
        if self.done():
            return '{} is done.'.format(self.job_id)
        else:
            return '{} is running.'.format(self.job_id)

    def stderr(self):
        if False:
            return 10
        return self.result()

    def stdout(self):
        if False:
            while True:
                i = 10
        return self.result()