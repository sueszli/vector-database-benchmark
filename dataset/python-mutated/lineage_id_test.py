import numpy as np
import syft as sy
from syft.types.uid import LineageID
from syft.types.uid import UID

def test_lineage_id() -> None:
    if False:
        for i in range(10):
            print('nop')
    data1 = sy.ActionObject.from_obj(syft_action_data=2 * np.random.rand(10, 10) - 1)
    data2 = sy.ActionObject.from_obj(syft_action_data=2 * np.random.rand(10, 10) - 1)
    left_lineage_id = data1.syft_lineage_id
    assert isinstance(left_lineage_id, LineageID)
    assert isinstance(left_lineage_id.id, UID)
    assert UID(left_lineage_id.id) == data1.id
    assert hash(data1.id) == left_lineage_id.syft_history_hash
    left_lineage_id2 = LineageID(value=data1.id, syft_history_hash=hash(data1.id))
    assert left_lineage_id == left_lineage_id2
    ser = sy.serialize(left_lineage_id, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)
    assert de == left_lineage_id
    result3 = data1 + data2
    result4 = data1 + data2
    assert result3.id != result4.id
    assert result3.syft_history_hash == result4.syft_history_hash
    left_child1 = result3 + data1
    right_child1 = result4 + data1
    assert left_child1.syft_history_hash == right_child1.syft_history_hash
    left_child2 = left_child1 - data1
    right_child2 = right_child1 + data1
    assert left_child2.syft_history_hash != right_child2.syft_history_hash