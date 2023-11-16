import unittest
import tempfile
from manticore.native import Manticore
from manticore.core.state import Concretize
from pathlib import Path
from glob import glob

class TestFork(unittest.TestCase):

    def test_fork_unique_solution(self):
        if False:
            i = 10
            return i + 15
        binary = str(Path(__file__).parent.parent.parent.joinpath('tests', 'native', 'binaries', 'hello_world'))
        tmp_dir = tempfile.TemporaryDirectory(prefix='mcore_test_fork_')
        m = Manticore(binary, stdin_size=10, workspace_url=str(tmp_dir.name))

        @m.hook(15952)
        def concretize_var(state):
            if False:
                for i in range(10):
                    print('nop')
            var = BitVecVariable(size=32, name='bar')
            state.constrain(var == 5)
            raise Concretize(var)
        m.run()
        m.finalize()
        states = f'{str(m.workspace)}/test_*.pkl'
        self.assertEqual(len(glob(states)), 1)
if __name__ == '__main__':
    unittest.main()