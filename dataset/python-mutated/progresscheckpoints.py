class ProgressCheckpoints:

    def __init__(self, num_jobs, num_checkpoints=10):
        if False:
            print('Hello World!')
        'Create a set of unique and evenly spaced indexes of jobs, used as checkpoints for progress'
        self.num_jobs = num_jobs
        self._checkpoints = {}
        if num_checkpoints > 0:
            self._offset = num_jobs / num_checkpoints
            for i in range(1, num_checkpoints):
                self._checkpoints[int(i * self._offset)] = 100 * i // num_checkpoints
            if num_jobs > 0:
                self._checkpoints[num_jobs - 1] = 100

    def is_checkpoint(self, index):
        if False:
            print('Hello World!')
        return index in self._checkpoints

    def progress(self, index):
        if False:
            return 10
        try:
            return self._checkpoints[index]
        except KeyError:
            return None