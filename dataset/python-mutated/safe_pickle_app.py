"""
This app tests three things
1. Can a work pickle `self`
2. Can the pickled work be unpickled in another work
3. Can the pickled work be unpickled from a script
"""
import subprocess
from pathlib import Path
from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.utilities import safe_pickle

class SelfPicklingWork(LightningWork):

    def run(self):
        if False:
            print('Hello World!')
        with open('work.pkl', 'wb') as f:
            safe_pickle.dump(self, f)

    def get_test_string(self):
        if False:
            i = 10
            return i + 15
        return f'Hello from {self.__class__.__name__}!'

class WorkThatLoadsPickledWork(LightningWork):

    def run(self):
        if False:
            i = 10
            return i + 15
        with open('work.pkl', 'rb') as f:
            work = safe_pickle.load(f)
        assert work.get_test_string() == 'Hello from SelfPicklingWork!'
script_load_pickled_work = '\nimport pickle\nwork = pickle.load(open("work.pkl", "rb"))\nprint(work.get_test_string())\n'

class RootFlow(LightningFlow):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.self_pickling_work = SelfPicklingWork()
        self.work_that_loads_pickled_work = WorkThatLoadsPickledWork()

    def run(self):
        if False:
            i = 10
            return i + 15
        self.self_pickling_work.run()
        self.work_that_loads_pickled_work.run()
        with open('script_that_loads_pickled_work.py', 'w') as f:
            f.write(script_load_pickled_work)
        proc = subprocess.Popen(['python', 'script_that_loads_pickled_work.py'], stdout=subprocess.PIPE)
        assert 'Hello from SelfPicklingWork' in proc.stdout.read().decode('UTF-8')
        Path('script_that_loads_pickled_work.py').unlink()
        Path('work.pkl').unlink()
        self.stop('Exiting the pickling app successfully!!')
app = LightningApp(RootFlow())