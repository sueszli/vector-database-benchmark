import glob
import os
import tarfile
from collections import Counter
from typing import Tuple
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader, ShardedDatasetReader
from allennlp.data.instance import Instance

def fingerprint(instance: Instance) -> Tuple[str, ...]:
    if False:
        return 10
    '\n    Get a hashable representation of a sequence tagging instance\n    that can be put in a Counter.\n    '
    text_tuple = tuple((t.text for t in instance.fields['tokens'].tokens))
    labels_tuple = tuple(instance.fields['tags'].labels)
    return text_tuple + labels_tuple

class TestShardedDatasetReader(AllenNlpTestCase):

    def setup_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setup_method()
        self.base_reader = SequenceTaggingDatasetReader(max_instances=4)
        base_file_path = AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv'
        raw_data = open(base_file_path).read()
        for i in range(100):
            file_path = self.TEST_DIR / f'identical_{i}.tsv'
            with open(file_path, 'w') as f:
                f.write(raw_data)
        self.identical_files_glob = str(self.TEST_DIR / 'identical_*.tsv')
        current_dir = os.getcwd()
        os.chdir(self.TEST_DIR)
        self.archive_filename = self.TEST_DIR / 'all_data.tar.gz'
        with tarfile.open(self.archive_filename, 'w:gz') as archive:
            for file_path in glob.glob('identical_*.tsv'):
                archive.add(file_path)
        os.chdir(current_dir)
        self.reader = ShardedDatasetReader(base_reader=self.base_reader)

    def read_and_check_instances(self, filepath: str, num_workers: int=0):
        if False:
            i = 10
            return i + 15
        data_loader = MultiProcessDataLoader(self.reader, filepath, num_workers=num_workers, batch_size=1, start_method='spawn')
        all_instances = []
        for instance in data_loader.iter_instances():
            all_instances.append(instance)
        assert len(all_instances) == 100 * 4
        counts = Counter((fingerprint(instance) for instance in all_instances))
        assert len(counts) == 4
        assert counts['cats', 'are', 'animals', '.', 'N', 'V', 'N', 'N'] == 100
        assert counts['dogs', 'are', 'animals', '.', 'N', 'V', 'N', 'N'] == 100
        assert counts['snakes', 'are', 'animals', '.', 'N', 'V', 'N', 'N'] == 100
        assert counts['birds', 'are', 'animals', '.', 'N', 'V', 'N', 'N'] == 100

    def test_sharded_read_glob(self):
        if False:
            print('Hello World!')
        self.read_and_check_instances(self.identical_files_glob)

    def test_sharded_read_with_multiprocess_loader(self):
        if False:
            print('Hello World!')
        self.read_and_check_instances(self.identical_files_glob, num_workers=2)

    def test_sharded_read_archive(self):
        if False:
            while True:
                i = 10
        self.read_and_check_instances(str(self.archive_filename))