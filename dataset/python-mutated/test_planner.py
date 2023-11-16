import sys
import torch
from torch.distributed.checkpoint.planner import LoadItemType, WriteItemType
from torch.distributed._shard.sharded_tensor import Shard, ShardMetadata, ShardedTensor, ShardedTensorMetadata
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.testing._internal.common_utils import TestCase, TEST_WITH_DEV_DBG_ASAN, run_tests
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, MetadataIndex, TensorStorageMetadata, ChunkStorageMetadata
from torch.testing._internal.distributed.distributed_utils import with_fake_comms, with_dist
from torch.distributed.checkpoint.default_planner import create_default_global_save_plan, create_default_local_save_plan, create_default_local_load_plan, _create_default_local_metadata
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list
from torch.distributed.checkpoint._dedup_tensors import dedup_tensors
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)

def create_sharded_tensor(rank, world_size, shards_per_rank, shard_size=8):
    if False:
        while True:
            i = 10
    shards_metadata = []
    local_shards = []
    for idx in range(0, world_size * shards_per_rank):
        shard_rank = idx // shards_per_rank
        shard_md = ShardMetadata(shard_offsets=[idx * shard_size], shard_sizes=[shard_size], placement=f'rank:{shard_rank}/cpu')
        shards_metadata.append(shard_md)
        if shard_rank == rank:
            shard = Shard.from_tensor_and_offsets(torch.rand(*shard_md.shard_sizes), shard_offsets=shard_md.shard_offsets, rank=rank)
            local_shards.append(shard)
    sharded_tensor_md = ShardedTensorMetadata(shards_metadata=shards_metadata, size=torch.Size([shard_size * len(shards_metadata)]), tensor_properties=TensorProperties.create_from_tensor(torch.zeros(1)))
    return ShardedTensor._init_from_local_shards_and_global_metadata(local_shards=local_shards, sharded_tensor_metadata=sharded_tensor_md)

class TestSavePlan(TestCase):

    @with_fake_comms(rank=1, world_size=4)
    def test_local_plan(self):
        if False:
            return 10
        tensor = torch.rand(10)
        val = [1, 2, 3]
        st = create_sharded_tensor(rank=1, world_size=4, shards_per_rank=1)
        state_dict = {'tensor': tensor, 'value': val, 'st': st}
        plan = create_default_local_save_plan(state_dict, False)
        self.assertEqual(2, len(plan.items))
        wi = plan.items[0]
        self.assertEqual(wi.index, MetadataIndex('tensor', [0]))
        self.assertEqual(wi.type, WriteItemType.TENSOR)
        self.assertEqual(wi.tensor_data.size, tensor.size())
        self.assertEqual(wi.tensor_data.properties, TensorProperties.create_from_tensor(torch.zeros(1)))
        self.assertEqual(wi.tensor_data.chunk.offsets, torch.Size([0]))
        self.assertEqual(wi.tensor_data.chunk.sizes, torch.Size([10]))
        st_wi = plan.items[1]
        self.assertEqual(st_wi.index, MetadataIndex('st', [8]))
        self.assertEqual(st_wi.type, WriteItemType.SHARD)
        self.assertEqual(st_wi.tensor_data.size, st.size())
        self.assertEqual(st_wi.tensor_data.properties, TensorProperties.create_from_tensor(torch.zeros(1)))
        self.assertEqual(st_wi.tensor_data.chunk.offsets, torch.Size([8]))
        self.assertEqual(st_wi.tensor_data.chunk.sizes, torch.Size([8]))
        plan = create_default_local_save_plan(state_dict, True)
        self.assertEqual(3, len(plan.items))
        tensor_wi = next((wi for wi in plan.items if wi.type == WriteItemType.TENSOR))
        self.assertEqual(tensor_wi.index, MetadataIndex('tensor', [0]))
        self.assertEqual(tensor_wi.tensor_data.size, tensor.size())
        self.assertEqual(tensor_wi.tensor_data.properties, TensorProperties.create_from_tensor(tensor))
        self.assertEqual(tensor_wi.tensor_data.chunk.offsets, torch.Size([0]))
        self.assertEqual(tensor_wi.tensor_data.chunk.sizes, torch.Size([10]))
        bytes_wi = next((wi for wi in plan.items if wi.type == WriteItemType.BYTE_IO))
        self.assertEqual(bytes_wi.index, MetadataIndex('value'))
        self.assertIsNone(bytes_wi.tensor_data)

    def test_global_plan(self):
        if False:
            return 10

        def create_data(rank):
            if False:
                return 10
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {'tensor': tensor, 'value': val, 'st': st}
                return create_default_local_save_plan(state_dict, rank == 0)
        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]
        all_plans = dedup_tensors(all_plans)
        (final_plans, metadata) = create_default_global_save_plan(all_plans=all_plans)
        for (new_plan, old_plan) in zip(final_plans, all_plans):
            for (new_item, old_item) in zip(new_plan.items, old_plan.items):
                self.assertEqual(new_item.index, old_item.index)
                self.assertEqual(new_item.type, old_item.type)
                self.assertEqual(new_item.tensor_data, old_item.tensor_data)
                self.assertIn(new_item.index.fqn, metadata.state_dict_metadata)
                item_md = metadata.state_dict_metadata[new_item.index.fqn]
                if new_item.type == WriteItemType.BYTE_IO:
                    self.assertTrue(isinstance(item_md, BytesStorageMetadata))
                else:
                    self.assertTrue(isinstance(item_md, TensorStorageMetadata))
                    self.assertEqual(item_md.size, old_item.tensor_data.size)
                    self.assertEqual(item_md.properties, old_item.tensor_data.properties)
                    self.assertIsNotNone(new_item.index.index)
                    self.assertEqual(item_md.chunks[new_item.index.index], old_item.tensor_data.chunk)

    def test_local_load_plan(self):
        if False:
            i = 10
            return i + 15

        def create_state_dict(rank):
            if False:
                return 10
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                return {'tensor': tensor, 'value': val, 'st': st}
        state_dict = create_state_dict(1)
        metadata = _create_default_local_metadata(state_dict)
        load_plan = create_default_local_load_plan(state_dict, metadata)
        self.assertEqual(3, len(load_plan.items))
        st_item = next((ri for ri in load_plan.items if ri.dest_index.fqn == 'st'))
        tensor_item = next((ri for ri in load_plan.items if ri.dest_index.fqn == 'tensor'))
        bytes_item = next((ri for ri in load_plan.items if ri.dest_index.fqn == 'value'))
        self.assertEqual(st_item.type, LoadItemType.TENSOR)
        self.assertEqual(st_item.dest_index, MetadataIndex('st', [8]))
        self.assertEqual(st_item.dest_offsets, torch.Size([0]))
        self.assertEqual(st_item.storage_index, MetadataIndex('st', [8]))
        self.assertEqual(st_item.storage_offsets, torch.Size([0]))
        self.assertEqual(st_item.lengths, torch.Size([8]))
        self.assertEqual(tensor_item.type, LoadItemType.TENSOR)
        self.assertEqual(tensor_item.dest_index, MetadataIndex('tensor', [0]))
        self.assertEqual(tensor_item.dest_offsets, torch.Size([0]))
        self.assertEqual(tensor_item.storage_index, MetadataIndex('tensor', [0]))
        self.assertEqual(tensor_item.storage_offsets, torch.Size([0]))
        self.assertEqual(tensor_item.lengths, torch.Size([10]))
        self.assertEqual(bytes_item.type, LoadItemType.BYTE_IO)
        self.assertEqual(bytes_item.dest_index, MetadataIndex('value'))

    def test_load_with_resharding(self):
        if False:
            for i in range(10):
                print('nop')

        def create_state_dict(rank, world_size):
            if False:
                print('Hello World!')
            with with_dist(rank=rank, world_size=world_size):
                return {'st': create_sharded_tensor(rank=rank, world_size=world_size, shards_per_rank=1, shard_size=128 // world_size)}
        world8_state_dict = create_state_dict(rank=1, world_size=8)
        world8_metadata = _create_default_local_metadata(world8_state_dict)
        world4_state_dict = create_state_dict(rank=1, world_size=4)
        world4_metadata = _create_default_local_metadata(world4_state_dict)
        load_plan = create_default_local_load_plan(world4_state_dict, world8_metadata)
        self.assertEqual(2, len(load_plan.items))
        low_ri = next((ri for ri in load_plan.items if ri.dest_offsets == torch.Size([0])))
        high_ri = next((ri for ri in load_plan.items if ri.dest_offsets == torch.Size([16])))
        self.assertEqual(low_ri.storage_index, MetadataIndex('st', [32]))
        self.assertEqual(low_ri.storage_offsets, torch.Size([0]))
        self.assertEqual(low_ri.dest_index, MetadataIndex('st', [32]))
        self.assertEqual(low_ri.dest_offsets, torch.Size([0]))
        self.assertEqual(low_ri.lengths, torch.Size([16]))
        self.assertEqual(high_ri.storage_index, MetadataIndex('st', [48]))
        self.assertEqual(high_ri.storage_offsets, torch.Size([0]))
        self.assertEqual(high_ri.dest_index, MetadataIndex('st', [32]))
        self.assertEqual(high_ri.dest_offsets, torch.Size([16]))
        self.assertEqual(high_ri.lengths, torch.Size([16]))
        load_plan = create_default_local_load_plan(world8_state_dict, world4_metadata)
        self.assertEqual(1, len(load_plan.items))
        ri = load_plan.items[0]
        self.assertEqual(ri.storage_index, MetadataIndex('st', [0]))
        self.assertEqual(ri.storage_offsets, torch.Size([16]))
        self.assertEqual(ri.dest_index, MetadataIndex('st', [16]))
        self.assertEqual(ri.dest_offsets, torch.Size([0]))
        self.assertEqual(ri.lengths, torch.Size([16]))

    def test_load_with_world_size_diff_by_one(self):
        if False:
            for i in range(10):
                print('nop')

        def create_state_dict(rank, world_size):
            if False:
                for i in range(10):
                    print('nop')
            with with_dist(rank=rank, world_size=world_size):
                return {'st': create_sharded_tensor(rank=rank, world_size=world_size, shards_per_rank=1, shard_size=120 // world_size)}
        world4_state_dict = create_state_dict(rank=1, world_size=4)
        world4_metadata = _create_default_local_metadata(world4_state_dict)
        world3_state_dict = create_state_dict(rank=1, world_size=3)
        load_plan = create_default_local_load_plan(world3_state_dict, world4_metadata)
        self.assertEqual(2, len(load_plan.items))
        low_ri = next((ri for ri in load_plan.items if ri.dest_offsets == torch.Size([0])))
        high_ri = next((ri for ri in load_plan.items if ri.dest_offsets == torch.Size([20])))
        self.assertEqual(low_ri.storage_index, MetadataIndex('st', [30]))
        self.assertEqual(low_ri.storage_offsets, torch.Size([10]))
        self.assertEqual(low_ri.dest_index, MetadataIndex('st', [40]))
        self.assertEqual(low_ri.dest_offsets, torch.Size([0]))
        self.assertEqual(low_ri.lengths, torch.Size([20]))
        self.assertEqual(high_ri.storage_index, MetadataIndex('st', [60]))
        self.assertEqual(high_ri.storage_offsets, torch.Size([0]))
        self.assertEqual(high_ri.dest_index, MetadataIndex('st', [40]))
        self.assertEqual(high_ri.dest_offsets, torch.Size([20]))
        self.assertEqual(high_ri.lengths, torch.Size([20]))

class TestPlannerHelpers(TestCase):

    def test_create_read_item_from_chunks(self):
        if False:
            for i in range(10):
                print('nop')
        tensor_md = TensorStorageMetadata(properties=TensorProperties.create_from_tensor(torch.empty([16])), size=torch.Size([16]), chunks=[ChunkStorageMetadata(offsets=torch.Size([0]), sizes=torch.Size([8])), ChunkStorageMetadata(offsets=torch.Size([8]), sizes=torch.Size([8]))])
        chunk = ChunkStorageMetadata(offsets=torch.Size([4]), sizes=torch.Size([7]))
        read_items = create_read_items_for_chunk_list('foo', tensor_md, [chunk])
        self.assertEqual(2, len(read_items))
        self.assertEqual(MetadataIndex('foo', [4]), read_items[0].dest_index)
        self.assertEqual(torch.Size([0]), read_items[0].dest_offsets)
        self.assertEqual(MetadataIndex('foo', [0]), read_items[0].storage_index)
        self.assertEqual(torch.Size([4]), read_items[0].storage_offsets)
        self.assertEqual(torch.Size([4]), read_items[0].lengths)
        self.assertEqual(MetadataIndex('foo', [4]), read_items[1].dest_index)
        self.assertEqual(torch.Size([4]), read_items[1].dest_offsets)
        self.assertEqual(MetadataIndex('foo', [8]), read_items[1].storage_index)
        self.assertEqual(torch.Size([0]), read_items[1].storage_offsets)
        self.assertEqual(torch.Size([3]), read_items[1].lengths)
if __name__ == '__main__':
    run_tests()