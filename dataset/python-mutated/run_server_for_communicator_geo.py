import os
from test_communicator_geo import TestCommunicatorGeoEnd2End
import paddle
paddle.enable_static()
pipe_name = os.getenv('PIPE_FILE')

class RunServer(TestCommunicatorGeoEnd2End):

    def runTest(self):
        if False:
            while True:
                i = 10
        pass
os.environ['TRAINING_ROLE'] = 'PSERVER'
half_run_server = RunServer()
with open(pipe_name, 'w') as pipe:
    pipe.write('done')
half_run_server.run_ut()