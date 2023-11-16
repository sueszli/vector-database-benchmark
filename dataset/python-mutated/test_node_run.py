import pytest
from kedro.io import LambdaDataset
from kedro.pipeline import node

@pytest.fixture
def mocked_dataset(mocker):
    if False:
        while True:
            i = 10
    load = mocker.Mock(return_value=42)
    save = mocker.Mock()
    return LambdaDataset(load, save)

def one_in_one_out(arg):
    if False:
        i = 10
        return i + 15
    return arg

def one_in_dict_out(arg):
    if False:
        i = 10
        return i + 15
    return {'ret': arg}

def two_in_first_out(arg1, arg2):
    if False:
        while True:
            i = 10
    return arg1

@pytest.fixture
def valid_nodes_with_inputs():
    if False:
        while True:
            i = 10
    return [(node(one_in_one_out, 'ds1', 'dsOut'), {'ds1': 42}), (node(one_in_dict_out, {'arg': 'ds1'}, {'ret': 'dsOut'}), {'ds1': 42}), (node(two_in_first_out, ['ds1', 'ds2'], 'dsOut'), {'ds1': 42, 'ds2': 58})]

def test_valid_nodes(valid_nodes_with_inputs):
    if False:
        i = 10
        return i + 15
    'Check if node.run works as expected.'
    for (node_, input_) in valid_nodes_with_inputs:
        output = node_.run(input_)
        assert output['dsOut'] == 42

def test_run_got_dataframe(mocked_dataset):
    if False:
        while True:
            i = 10
    'Check an exception when non-dictionary (class object) is passed.'
    pattern = 'Node.run\\(\\) expects a dictionary or None, '
    pattern += "but got <class \\'kedro.io.lambda_dataset.LambdaDataset\\'> instead"
    with pytest.raises(ValueError, match=pattern):
        node(one_in_one_out, {'arg': 'ds1'}, 'A').run(mocked_dataset)

class TestNodeRunInvalidInput:

    def test_unresolved(self):
        if False:
            i = 10
            return i + 15
        'Pass no input when one is expected.'
        with pytest.raises(ValueError, match='expected one input'):
            node(one_in_one_out, 'unresolved', 'ds1').run(None)

    def test_no_inputs_node_error(self, mocked_dataset):
        if False:
            print('Hello World!')
        'Pass one input when none is expected.'
        with pytest.raises(ValueError, match='expected no inputs'):
            node(lambda : 1, None, 'A').run({'unexpected': mocked_dataset})

    def test_one_input_error(self, mocked_dataset):
        if False:
            return 10
        'Pass a different input.'
        pattern = "expected one input named 'ds1', but got the "
        pattern += "following 1 input\\(s\\) instead: \\['arg'\\]"
        with pytest.raises(ValueError, match=pattern):
            node(one_in_dict_out, 'ds1', {'ret': 'B', 'ans': 'C'}).run({'arg': mocked_dataset})

    def test_run_diff_size_lists(self, mocked_dataset):
        if False:
            return 10
        'Pass only one dict input when two (list) are expected.'
        pattern = "expected 2 input\\(s\\) \\['ds1', 'ds2'\\], but "
        pattern += 'got the following 1 input\\(s\\) instead.'
        with pytest.raises(ValueError, match=pattern):
            node(two_in_first_out, ['ds1', 'ds2'], 'A').run({'ds1': mocked_dataset})

    def test_run_diff_size_list_dict(self, mocked_dataset):
        if False:
            print('Hello World!')
        'Pass two dict inputs when one (list) are expected.'
        pattern = "expected 1 input\\(s\\) \\['ds1'\\], but got the "
        pattern += "following 2 input\\(s\\) instead: \\['ds1', 'ds2'\\]\\."
        with pytest.raises(ValueError, match=pattern):
            node(one_in_one_out, ['ds1'], 'A').run({'ds1': mocked_dataset, 'ds2': 2})

    def test_run_list_dict_unavailable(self, mocked_dataset):
        if False:
            while True:
                i = 10
        'Pass one dict which is different from expected.'
        pattern = "expected 1 input\\(s\\) \\['ds1'\\], but got the "
        pattern += "following 1 input\\(s\\) instead: \\['ds2'\\]\\."
        with pytest.raises(ValueError, match=pattern):
            node(one_in_one_out, ['ds1'], 'A').run({'ds2': mocked_dataset})

    def test_run_dict_unavailable(self, mocked_dataset):
        if False:
            i = 10
            return i + 15
        'Pass one dict which is different from expected.'
        pattern = "expected 1 input\\(s\\) \\['ds1'\\], but got the "
        pattern += "following 1 input\\(s\\) instead: \\['ds2'\\]\\."
        with pytest.raises(ValueError, match=pattern):
            node(one_in_one_out, {'arg': 'ds1'}, 'A').run({'ds2': mocked_dataset})

    def test_run_dict_diff_size(self, mocked_dataset):
        if False:
            print('Hello World!')
        'Pass two dict inputs when one is expected.'
        pattern = "expected 1 input\\(s\\) \\['ds1'\\], but got the "
        pattern += "following 2 input\\(s\\) instead: \\['ds1', 'ds2'\\]\\."
        with pytest.raises(ValueError, match=pattern):
            node(one_in_one_out, {'arg': 'ds1'}, 'A').run({'ds1': mocked_dataset, 'ds2': 2})

class TestNodeRunInvalidOutput:

    def test_miss_matching_output_types(self, mocked_dataset):
        if False:
            while True:
                i = 10
        pattern = 'The node output is a dictionary, whereas the function '
        pattern += "output is <class 'kedro.io.lambda_dataset.LambdaDataset'>."
        with pytest.raises(ValueError, match=pattern):
            node(one_in_one_out, 'ds1', {'a': 'ds'}).run({'ds1': mocked_dataset})

    def test_miss_matching_output_keys(self, mocked_dataset):
        if False:
            for i in range(10):
                print('nop')
        pattern = "The node's output keys {'ret'} do not match "
        pattern += "with the returned output's keys"
        with pytest.raises(ValueError, match=pattern):
            node(one_in_dict_out, 'ds1', {'ret': 'B', 'ans': 'C'}).run({'ds1': mocked_dataset})

    def test_node_not_list_output(self, mocked_dataset):
        if False:
            for i in range(10):
                print('nop')
        pattern = 'The node definition contains a list of outputs '
        pattern += "\\['B', 'C'\\], whereas the node function returned "
        pattern += "a 'LambdaDataset'"
        with pytest.raises(ValueError, match=pattern):
            node(one_in_one_out, 'ds1', ['B', 'C']).run({'ds1': mocked_dataset})

    def test_node_wrong_num_of_outputs(self, mocker, mocked_dataset):
        if False:
            i = 10
            return i + 15

        def one_in_two_out(arg):
            if False:
                print('Hello World!')
            load = mocker.Mock(return_value=42)
            save = mocker.Mock()
            return [LambdaDataset(load, save), LambdaDataset(load, save)]
        pattern = 'The node function returned 2 output\\(s\\), whereas '
        pattern += 'the node definition contains 3 output\\(s\\)\\.'
        with pytest.raises(ValueError, match=pattern):
            node(one_in_two_out, 'ds1', ['A', 'B', 'C']).run({'ds1': mocked_dataset})