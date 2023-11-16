"""Library to run Jackhmmer from Python."""
import glob
import os
import subprocess
from concurrent import futures
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib import request
from absl import logging
from . import utils

class Jackhmmer:
    """Python wrapper of the Jackhmmer binary."""

    def __init__(self, *, binary_path: str, database_path: str, n_cpu: int=8, n_iter: int=1, e_value: float=0.0001, z_value: Optional[int]=None, get_tblout: bool=False, filter_f1: float=0.0005, filter_f2: float=5e-05, filter_f3: float=5e-07, incdom_e: Optional[float]=None, dom_e: Optional[float]=None, num_streamed_chunks: Optional[int]=None, streaming_callback: Optional[Callable[[int], None]]=None):
        if False:
            i = 10
            return i + 15
        'Initializes the Python Jackhmmer wrapper.\n\n        Args:\n            binary_path: The path to the jackhmmer executable.\n            database_path: The path to the jackhmmer database (FASTA format).\n            n_cpu: The number of CPUs to give Jackhmmer.\n            n_iter: The number of Jackhmmer iterations.\n            e_value: The E-value, see Jackhmmer docs for more details.\n            z_value: The Z-value, see Jackhmmer docs for more details.\n            get_tblout: Whether to save tblout string.\n            filter_f1: MSV and biased composition pre-filter, set to >1.0 to turn off.\n            filter_f2: Viterbi pre-filter, set to >1.0 to turn off.\n            filter_f3: Forward pre-filter, set to >1.0 to turn off.\n            incdom_e: Domain e-value criteria for inclusion of domains in MSA/next\n                round.\n            dom_e: Domain e-value criteria for inclusion in tblout.\n            num_streamed_chunks: Number of database chunks to stream over.\n            streaming_callback: Callback function run after each chunk iteration with\n                the iteration number as argument.\n        '
        self.binary_path = binary_path
        self.database_path = database_path
        self.num_streamed_chunks = num_streamed_chunks
        if not os.path.exists(self.database_path) and num_streamed_chunks is None:
            logging.error('Could not find Jackhmmer database %s', database_path)
            raise ValueError(f'Could not find Jackhmmer database {database_path}')
        self.n_cpu = n_cpu
        self.n_iter = n_iter
        self.e_value = e_value
        self.z_value = z_value
        self.filter_f1 = filter_f1
        self.filter_f2 = filter_f2
        self.filter_f3 = filter_f3
        self.incdom_e = incdom_e
        self.dom_e = dom_e
        self.get_tblout = get_tblout
        self.streaming_callback = streaming_callback

    def _query_chunk(self, input_fasta_path: str, database_path: str) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        'Queries the database chunk using Jackhmmer.'
        with utils.tmpdir_manager() as query_tmp_dir:
            sto_path = os.path.join(query_tmp_dir, 'output.sto')
            cmd_flags = ['-o', '/dev/null', '-A', sto_path, '--noali', '--F1', str(self.filter_f1), '--F2', str(self.filter_f2), '--F3', str(self.filter_f3), '--incE', str(self.e_value), '-E', str(self.e_value), '--cpu', str(self.n_cpu), '-N', str(self.n_iter)]
            if self.get_tblout:
                tblout_path = os.path.join(query_tmp_dir, 'tblout.txt')
                cmd_flags.extend(['--tblout', tblout_path])
            if self.z_value:
                cmd_flags.extend(['-Z', str(self.z_value)])
            if self.dom_e is not None:
                cmd_flags.extend(['--domE', str(self.dom_e)])
            if self.incdom_e is not None:
                cmd_flags.extend(['--incdomE', str(self.incdom_e)])
            cmd = [self.binary_path] + cmd_flags + [input_fasta_path, database_path]
            logging.info('Launching subprocess "%s"', ' '.join(cmd))
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with utils.timing(f'Jackhmmer ({os.path.basename(database_path)}) query'):
                (_, stderr) = process.communicate()
                retcode = process.wait()
            if retcode:
                raise RuntimeError('Jackhmmer failed\nstderr:\n%s\n' % stderr.decode('utf-8'))
            tbl = ''
            if self.get_tblout:
                with open(tblout_path) as f:
                    tbl = f.read()
            with open(sto_path) as f:
                sto = f.read()
        raw_output = dict(sto=sto, tbl=tbl, stderr=stderr, n_iter=self.n_iter, e_value=self.e_value)
        return raw_output

    def query(self, input_fasta_path: str) -> Sequence[Mapping[str, Any]]:
        if False:
            print('Hello World!')
        'Queries the database using Jackhmmer.'
        if self.num_streamed_chunks is None:
            return [self._query_chunk(input_fasta_path, self.database_path)]
        db_basename = os.path.basename(self.database_path)

        def db_remote_chunk(db_idx):
            if False:
                i = 10
                return i + 15
            return f'{self.database_path}.{db_idx}'

        def db_local_chunk(db_idx):
            if False:
                while True:
                    i = 10
            return f'/tmp/ramdisk/{db_basename}.{db_idx}'
        for f in glob.glob(db_local_chunk('[0-9]*')):
            try:
                os.remove(f)
            except OSError:
                print(f'OSError while deleting {f}')
        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            chunked_output = []
            for i in range(1, self.num_streamed_chunks + 1):
                if i == 1:
                    future = executor.submit(request.urlretrieve, db_remote_chunk(i), db_local_chunk(i))
                if i < self.num_streamed_chunks:
                    next_future = executor.submit(request.urlretrieve, db_remote_chunk(i + 1), db_local_chunk(i + 1))
                future.result()
                chunked_output.append(self._query_chunk(input_fasta_path, db_local_chunk(i)))
                os.remove(db_local_chunk(i))
                if i < self.num_streamed_chunks:
                    future = next_future
                if self.streaming_callback:
                    self.streaming_callback(i)
        return chunked_output