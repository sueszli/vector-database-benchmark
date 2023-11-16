from typing import List
import pytest
from syft.service.dataset.dataset import Dataset
from syft.service.dataset.dataset_stash import ActionIDsPartitionKey
from syft.service.dataset.dataset_stash import NamePartitionKey
from syft.store.document_store import QueryKey
from syft.types.uid import UID

def test_dataset_namepartitionkey() -> None:
    if False:
        while True:
            i = 10
    mock_obj = 'dummy_name_key'
    assert NamePartitionKey.key == 'name'
    assert NamePartitionKey.type_ == str
    name_partition_key = NamePartitionKey.with_obj(obj=mock_obj)
    assert isinstance(name_partition_key, QueryKey)
    assert name_partition_key.key == 'name'
    assert name_partition_key.type_ == str
    assert name_partition_key.value == mock_obj
    with pytest.raises(AttributeError):
        NamePartitionKey.with_obj(obj=[UID()])

def test_dataset_actionidpartitionkey() -> None:
    if False:
        while True:
            i = 10
    mock_obj = [UID() for _ in range(3)]
    assert ActionIDsPartitionKey.key == 'action_ids'
    assert ActionIDsPartitionKey.type_ == List[UID]
    action_ids_partition_key = ActionIDsPartitionKey.with_obj(obj=mock_obj)
    assert isinstance(action_ids_partition_key, QueryKey)
    assert action_ids_partition_key.key == 'action_ids'
    assert action_ids_partition_key.type_ == List[UID]
    assert action_ids_partition_key.value == mock_obj
    with pytest.raises(AttributeError):
        ActionIDsPartitionKey.with_obj(obj='dummy_str')
    with pytest.raises(TypeError):
        ActionIDsPartitionKey.with_obj(obj=['first_str', 'second_str'])

def test_dataset_get_by_name(root_verify_key, mock_dataset_stash, mock_dataset) -> None:
    if False:
        print('Hello World!')
    result = mock_dataset_stash.get_by_name(root_verify_key, mock_dataset.name)
    assert result.is_ok(), f'Dataset could not be retrieved, result: {result}'
    assert isinstance(result.ok(), Dataset)
    assert result.ok().id == mock_dataset.id
    result = mock_dataset_stash.get_by_name(root_verify_key, 'non_existing_dataset')
    assert result.is_ok(), f'Dataset could not be retrieved, result: {result}'
    assert result.ok() is None

@pytest.mark.xfail(raises=AttributeError, reason='DatasetUpdate is not implemeted yet')
def test_dataset_update(root_verify_key, mock_dataset_stash, mock_dataset, mock_dataset_update) -> None:
    if False:
        return 10
    result = mock_dataset_stash.update(root_verify_key, dataset_update=mock_dataset_update)
    assert result.is_ok(), f'Dataset could not be retrieved, result: {result}'
    assert isinstance(result.ok(), Dataset)
    assert mock_dataset.id == result.ok().id
    other_obj = object()
    result = mock_dataset_stash.update(root_verify_key, dataset_update=other_obj)
    assert result.err(), f'Dataset was updated with non-DatasetUpdate object,result: {result}'

def test_dataset_search_action_ids(root_verify_key, mock_dataset_stash, mock_dataset):
    if False:
        for i in range(10):
            print('nop')
    action_id = mock_dataset.assets[0].action_id
    result = mock_dataset_stash.search_action_ids(root_verify_key, uid=action_id)
    assert result.is_ok(), f'Dataset could not be retrieved, result: {result}'
    assert result.ok() != [], f'Dataset was not found by action_id {action_id}'
    assert isinstance(result.ok()[0], Dataset)
    assert result.ok()[0].id == mock_dataset.id
    result = mock_dataset_stash.search_action_ids(root_verify_key, uid=[action_id])
    assert result.is_ok(), f'Dataset could not be retrieved, result: {result}'
    assert isinstance(result.ok()[0], Dataset)
    assert result.ok()[0].id == mock_dataset.id
    other_action_id = UID()
    result = mock_dataset_stash.search_action_ids(root_verify_key, uid=other_action_id)
    assert result.is_ok(), f'Dataset could not be retrieved, result: {result}'
    assert result.ok() == []
    random_obj = object()
    with pytest.raises(AttributeError):
        result = mock_dataset_stash.search_action_ids(root_verify_key, uid=random_obj)