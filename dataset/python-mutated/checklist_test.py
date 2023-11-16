import argparse
import sys
from allennlp.commands import main
from allennlp.commands.checklist import CheckList
from allennlp.common.testing import AllenNlpTestCase, requires_gpu

class TestCheckList(AllenNlpTestCase):

    def setup_method(self):
        if False:
            return 10
        super().setup_method()
        self.archive_file = self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz'
        self.task = 'sentiment-analysis'

    def test_add_checklist_subparser(self):
        if False:
            i = 10
            return i + 15
        parser = argparse.ArgumentParser(description='Testing')
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        CheckList().add_subparser(subparsers)
        kebab_args = ['checklist', '/path/to/archive', 'task-suite-name', '--checklist-suite', '/path/to/checklist/pkl', '--output-file', '/dev/null', '--cuda-device', '0']
        args = parser.parse_args(kebab_args)
        assert args.func.__name__ == '_run_suite'
        assert args.archive_file == '/path/to/archive'
        assert args.task == 'task-suite-name'
        assert args.output_file == '/dev/null'
        assert args.cuda_device == 0

    @requires_gpu
    def test_works_with_known_model(self):
        if False:
            for i in range(10):
                print('nop')
        sys.argv = ['__main__.py', 'checklist', str(self.archive_file), str(self.task), '--task-suite-args', '{"positive": 1, "negative": 0}', '--max-examples', '1', '--cuda-device', '0']
        main()