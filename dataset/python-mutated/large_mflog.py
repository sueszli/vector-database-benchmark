from metaflow_test import MetaflowTest, ExpectationFailed, steps

class LargeMflogTest(MetaflowTest):
    """
    Test that we can capture a large amount of log messages with
    accurate timings
    """
    PRIORITY = 2
    HEADER = '\nNUM_FOREACH = 32\nNUM_LINES = 5000\n'

    @steps(0, ['foreach-split-small'], required=True)
    def split(self):
        if False:
            i = 10
            return i + 15
        self.arr = range(NUM_FOREACH)
        import random
        import string
        self.random_log_prefix = ''.join([random.choice(string.ascii_lowercase) for _ in range(5)])

    @steps(0, ['foreach-inner-small'], required=True)
    def inner(self):
        if False:
            i = 10
            return i + 15
        ISOFORMAT = '%Y-%m-%dT%H:%M:%S.%f'
        from datetime import datetime
        from metaflow import current
        import sys
        self.log_step = current.step_name
        task_id = current.task_id
        for i in range(NUM_LINES):
            now = datetime.utcnow().strftime(ISOFORMAT)
            print('%s %s stdout %d %s' % (self.random_log_prefix, task_id, i, now))
            sys.stderr.write('%s %s stderr %d %s\n' % (self.random_log_prefix, task_id, i, now))

    @steps(0, ['foreach-join-small'], required=True)
    def join(self, inputs):
        if False:
            print('Hello World!')
        self.log_step = inputs[0].log_step
        self.random_log_prefix = inputs[0].random_log_prefix

    @steps(1, ['all'])
    def step_all(self):
        if False:
            while True:
                i = 10
        pass

    @steps(0, ['end'])
    def step_end(self):
        if False:
            while True:
                i = 10
        self.num_foreach = NUM_FOREACH
        self.num_lines = NUM_LINES

    def check_results(self, flow, checker):
        if False:
            return 10
        from itertools import groupby
        from datetime import datetime
        ISOFORMAT = '%Y-%m-%dT%H:%M:%S.%f'
        _val = lambda n: list(checker.artifact_dict('end', n).values())[0][n]
        step_name = _val('log_step')
        num_foreach = _val('num_foreach')
        num_lines = _val('num_lines')
        random_log_prefix = _val('random_log_prefix')
        run = checker.get_run()
        for stream in ('stdout', 'stderr'):
            log = checker.get_log(step_name, stream)
            lines = [line.split() for line in log.splitlines() if line.startswith(random_log_prefix)]
            assert_equals(len(lines), num_foreach * num_lines)
            for (task_id, task_lines_iter) in groupby(lines, lambda x: x[1]):
                task_lines = list(task_lines_iter)
                assert_equals(len(task_lines), num_lines)
                for (i, (_, _, stream_type, idx, tstamp)) in enumerate(task_lines):
                    assert_equals(stream_type, stream)
                    assert_equals(int(idx), i)
            if run is not None:
                for task in run[step_name]:
                    task_lines = [(tstamp, msg) for (tstamp, msg) in task.loglines(stream) if msg.startswith(random_log_prefix)]
                    assert_equals(len(task_lines), num_lines)
                    for (i, (mf_tstamp, msg)) in enumerate(task_lines):
                        (_, task_id, stream_type, idx, tstamp_str) = msg.split()
                        assert_equals(task_id, task.id)
                        assert_equals(stream_type, stream)
                        assert_equals(int(idx), i)