import pytest
from airbyte_cdk.sources.streams.core import Stream
from source_trello.components import OrderIdsPartitionRouter

class MockStream(Stream):

    def __init__(self, records):
        if False:
            while True:
                i = 10
        self.records = records

    def primary_key(self):
        if False:
            print('Hello World!')
        return

    def read_records(self, sync_mode):
        if False:
            for i in range(10):
                print('nop')
        return self.records
test_cases = [([{'id': 'b11111111111111111111111', 'name': 'board_1'}, {'id': 'b22222222222222222222222', 'name': 'board_2'}], [{'id': 'org111111111111111111111', 'idBoards': ['b11111111111111111111111', 'b22222222222222222222222']}], ['b11111111111111111111111', 'b22222222222222222222222']), ([{'id': 'b11111111111111111111111', 'name': 'board_1'}, {'id': 'b22222222222222222222222', 'name': 'board_2'}], [{'id': 'org111111111111111111111', 'idBoards': ['b11111111111111111111111', 'b33333333333333333333333']}], ['b11111111111111111111111', 'b22222222222222222222222', 'b33333333333333333333333']), ([{'id': 'b11111111111111111111111', 'name': 'board_1'}, {'id': 'b22222222222222222222222', 'name': 'board_2'}], [{'id': 'org111111111111111111111', 'idBoards': ['b11111111111111111111111', 'b33333333333333333333333']}, {'id': 'org222222222222222222222', 'idBoards': ['b00000000000000000000000', 'b44444444444444444444444']}], ['b11111111111111111111111', 'b22222222222222222222222', 'b33333333333333333333333', 'b00000000000000000000000', 'b44444444444444444444444']), ([], [], [])]

@pytest.mark.parametrize('boards_records, organizations_records, expected_board_ids', test_cases)
def test_read_all_boards(boards_records, organizations_records, expected_board_ids):
    if False:
        i = 10
        return i + 15
    partition_router = OrderIdsPartitionRouter(parent_stream_configs=[None], config=None, parameters=None)
    boards_stream = MockStream(records=boards_records)
    organizations_stream = MockStream(records=organizations_records)
    board_ids = list(partition_router.read_all_boards(boards_stream, organizations_stream))
    assert board_ids == expected_board_ids