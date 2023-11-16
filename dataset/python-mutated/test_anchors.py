"""
CommandLine:
    pytest tests/test_utils/test_anchor.py
    xdoctest tests/test_utils/test_anchor.py zero

"""
import torch
from mmdet3d.core.anchor import build_prior_generator

def test_anchor_3d_range_generator():
    if False:
        return 10
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    anchor_generator_cfg = dict(type='Anchor3DRangeGenerator', ranges=[[0, -39.68, -0.6, 70.4, 39.68, -0.6], [0, -39.68, -0.6, 70.4, 39.68, -0.6], [0, -39.68, -1.78, 70.4, 39.68, -1.78]], sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]], rotations=[0, 1.57], reshape_out=False)
    anchor_generator = build_prior_generator(anchor_generator_cfg)
    repr_str = repr(anchor_generator)
    expected_repr_str = 'Anchor3DRangeGenerator(anchor_range=[[0, -39.68, -0.6, 70.4, 39.68, -0.6], [0, -39.68, -0.6, 70.4, 39.68, -0.6], [0, -39.68, -1.78, 70.4, 39.68, -1.78]],\nscales=[1],\nsizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],\nrotations=[0, 1.57],\nreshape_out=False,\nsize_per_range=True)'
    assert repr_str == expected_repr_str
    featmap_size = (256, 256)
    mr_anchors = anchor_generator.single_level_grid_anchors(featmap_size, 1.1, device=device)
    assert mr_anchors.shape == torch.Size([1, 256, 256, 3, 2, 7])

def test_aligned_anchor_generator():
    if False:
        return 10
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    anchor_generator_cfg = dict(type='AlignedAnchor3DRangeGenerator', ranges=[[-51.2, -51.2, -1.8, 51.2, 51.2, -1.8]], scales=[1, 2, 4], sizes=[[2.5981, 0.866, 1.0], [1.7321, 0.5774, 1.0], [1.0, 1.0, 1.0], [0.4, 0.4, 1]], custom_values=[0, 0], rotations=[0, 1.57], size_per_range=False, reshape_out=True)
    featmap_sizes = [(256, 256), (128, 128), (64, 64)]
    anchor_generator = build_prior_generator(anchor_generator_cfg)
    assert anchor_generator.num_base_anchors == 8
    expected_grid_anchors = [torch.tensor([[-51.0, -51.0, -1.8, 2.5981, 0.866, 1.0, 0.0, 0.0, 0.0], [-51.0, -51.0, -1.8, 0.4, 0.4, 1.0, 1.57, 0.0, 0.0], [-50.6, -51.0, -1.8, 0.4, 0.4, 1.0, 0.0, 0.0, 0.0], [-50.2, -51.0, -1.8, 1.0, 1.0, 1.0, 1.57, 0.0, 0.0], [-49.8, -51.0, -1.8, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [-49.4, -51.0, -1.8, 1.7321, 0.5774, 1.0, 1.57, 0.0, 0.0], [-49.0, -51.0, -1.8, 1.7321, 0.5774, 1.0, 0.0, 0.0, 0.0], [-48.6, -51.0, -1.8, 2.5981, 0.866, 1.0, 1.57, 0.0, 0.0]], device=device), torch.tensor([[-50.8, -50.8, -1.8, 5.1962, 1.732, 2.0, 0.0, 0.0, 0.0], [-50.8, -50.8, -1.8, 0.8, 0.8, 2.0, 1.57, 0.0, 0.0], [-50.0, -50.8, -1.8, 0.8, 0.8, 2.0, 0.0, 0.0, 0.0], [-49.2, -50.8, -1.8, 2.0, 2.0, 2.0, 1.57, 0.0, 0.0], [-48.4, -50.8, -1.8, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0], [-47.6, -50.8, -1.8, 3.4642, 1.1548, 2.0, 1.57, 0.0, 0.0], [-46.8, -50.8, -1.8, 3.4642, 1.1548, 2.0, 0.0, 0.0, 0.0], [-46.0, -50.8, -1.8, 5.1962, 1.732, 2.0, 1.57, 0.0, 0.0]], device=device), torch.tensor([[-50.4, -50.4, -1.8, 10.3924, 3.464, 4.0, 0.0, 0.0, 0.0], [-50.4, -50.4, -1.8, 1.6, 1.6, 4.0, 1.57, 0.0, 0.0], [-48.8, -50.4, -1.8, 1.6, 1.6, 4.0, 0.0, 0.0, 0.0], [-47.2, -50.4, -1.8, 4.0, 4.0, 4.0, 1.57, 0.0, 0.0], [-45.6, -50.4, -1.8, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0], [-44.0, -50.4, -1.8, 6.9284, 2.3096, 4.0, 1.57, 0.0, 0.0], [-42.4, -50.4, -1.8, 6.9284, 2.3096, 4.0, 0.0, 0.0, 0.0], [-40.8, -50.4, -1.8, 10.3924, 3.464, 4.0, 1.57, 0.0, 0.0]], device=device)]
    multi_level_anchors = anchor_generator.grid_anchors(featmap_sizes, device=device)
    expected_multi_level_shapes = [torch.Size([524288, 9]), torch.Size([131072, 9]), torch.Size([32768, 9])]
    for (i, single_level_anchor) in enumerate(multi_level_anchors):
        assert single_level_anchor.shape == expected_multi_level_shapes[i]
        assert single_level_anchor[:56:7].allclose(expected_grid_anchors[i])

def test_aligned_anchor_generator_per_cls():
    if False:
        print('Hello World!')
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    anchor_generator_cfg = dict(type='AlignedAnchor3DRangeGeneratorPerCls', ranges=[[-100, -100, -1.8, 100, 100, -1.8], [-100, -100, -1.3, 100, 100, -1.3]], sizes=[[1.76, 0.63, 1.44], [2.35, 0.96, 1.59]], custom_values=[0, 0], rotations=[0, 1.57], reshape_out=False)
    featmap_sizes = [(100, 100), (50, 50)]
    anchor_generator = build_prior_generator(anchor_generator_cfg)
    expected_grid_anchors = [[torch.tensor([[-99.0, -99.0, -1.8, 1.76, 0.63, 1.44, 0.0, 0.0, 0.0], [-99.0, -99.0, -1.8, 1.76, 0.63, 1.44, 1.57, 0.0, 0.0]], device=device), torch.tensor([[-98.0, -98.0, -1.3, 2.35, 0.96, 1.59, 0.0, 0.0, 0.0], [-98.0, -98.0, -1.3, 2.35, 0.96, 1.59, 1.57, 0.0, 0.0]], device=device)]]
    multi_level_anchors = anchor_generator.grid_anchors(featmap_sizes, device=device)
    expected_multi_level_shapes = [[torch.Size([20000, 9]), torch.Size([5000, 9])]]
    for (i, single_level_anchor) in enumerate(multi_level_anchors):
        assert len(single_level_anchor) == len(expected_multi_level_shapes[i])
        for j in range(len(single_level_anchor)):
            interval = int(expected_multi_level_shapes[i][j][0] / 2)
            assert single_level_anchor[j][:2 * interval:interval].allclose(expected_grid_anchors[i][j])