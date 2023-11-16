from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import assert_op_count_match, assert_model_is_valid, get_op_types_in_program, apply_pass_and_basic_check
import unittest
import pytest
import numpy as np
np.random.seed(1984)

class TransposeOptimizationPass(unittest.TestCase):
    """"""
    '\n    Input graph:\n    input -----> transpose(axis=[1,0]) -----> transpose(axis=[1,0]) ----> relu ---> out\n\n    Output graph:\n    input -----> relu -----> out\n    '

    def test_simple_consecutive_ops_fusion(self):
        if False:
            for i in range(10):
                print('nop')

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            if False:
                i = 10
                return i + 15
            x = mb.transpose(x=x, perm=[1, 0])
            x = mb.transpose(x=x, perm=[1, 0])
            x = mb.relu(x=x)
            return x
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'transpose', 'relu'])
        self.assertEqual(get_op_types_in_program(prog), ['relu'])
        assert_model_is_valid(prog, {'x': (10, 20)}, expected_output_shapes={block.outputs[0].name: (10, 20)})
    '\n    Input graph:\n    input---->transpose(axis=[0,3,1,2])---->relu---->log--->transpose(axis=[0,2,3,1])--->relu--->out\n\n    Output graph:\n    input----->relu----->log----->relu--->out\n    '

    def test_linear_graph_two_op_fusion(self):
        if False:
            i = 10
            return i + 15

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x):
            if False:
                i = 10
                return i + 15
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x = mb.relu(x=x)
            x = mb.log(x=x)
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x = mb.relu(x=x)
            return x
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'log', 'transpose', 'relu'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'log', 'relu'])
        assert_model_is_valid(prog, {'x': (1, 2, 3, 4)}, expected_output_shapes={block.outputs[0].name: (1, 2, 3, 4)})
    '\n    Input graph:\n    input---->transpose(axis=[0,3,1,2])---->relu---->identity--->transpose(axis=[0,2,3,1])--->relu--->out\n\n    Output graph:\n    input----->relu----->identity----->relu--->out\n    '

    def test_linear_graph_two_op_fusion_1(self):
        if False:
            while True:
                i = 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x):
            if False:
                for i in range(10):
                    print('nop')
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x = mb.relu(x=x)
            x = mb.identity(x=x)
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x = mb.relu(x=x)
            return x
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'identity', 'transpose', 'relu'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'identity', 'relu'])
        assert_model_is_valid(prog, {'x': (1, 2, 3, 4)}, expected_output_shapes={block.outputs[0].name: (1, 2, 3, 4)})
    '\n    Input graph:\n    input(shape=1,2,3,4)---->transpose(axis=[0,3,1,2])---->relu---->log--->transpose(axis=[0,2,3,1])--->relu--->out1(shape=1,2,3,4)\n                                                                |\n                                                                v\n                                                        out2(shape=1,4,2,3)\n\n    Output graph:\n    input(shape=1,2,3,4)---->relu---->log--->relu--->out1(shape=1,2,3,4)\n                                  |\n                                  |----->transpose(axis=[0,3,1,2])----->out2(shape=1,4,2,3)\n    '

    def test_fusion_with_output_edge_inbetween(self):
        if False:
            print('Hello World!')

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x):
            if False:
                return 10
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x1 = mb.relu(x=x)
            x2 = mb.log(x=x1)
            x3 = mb.transpose(x=x2, perm=[0, 2, 3, 1])
            x4 = mb.relu(x=x3)
            return (x4, x1)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'log', 'transpose', 'relu'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'log', 'relu', 'transpose'])
        assert_model_is_valid(prog, {'x': (1, 2, 3, 4)}, expected_output_shapes={block.outputs[0].name: (1, 2, 3, 4), block.outputs[1].name: (1, 4, 2, 3)})
    '\n    Input graph:\n    input---->transpose(axis=[0,3,1,2])---->relu---->transpose(axis=[0,2,3,1])--->out\n\n    Output graph:\n    input----->relu----->out\n    '

    def test_linear_graph_two_op_fusion_with_last_op_removal(self):
        if False:
            while True:
                i = 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x):
            if False:
                print('Hello World!')
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x = mb.relu(x=x)
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            return x
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu'])
        assert_model_is_valid(prog, {'x': (1, 2, 3, 4)}, expected_output_shapes={block.outputs[0].name: (1, 2, 3, 4)})
    '\n    Input graph:\n    input(shape=10,2,3)--->transpose(axis=[0,2,1])----->relu---->transpose(axis=[0,2,1])---->out1\n                                                    |\n                                                    |\n                                                    --->relu----->log---->transpose(axis=[0,2,1])---->out2\n\n    Output graph:\n    input(shape=10,2,3)----->relu---->out1\n                        |\n                        |\n                        --->relu----->log---->out2\n    '

    def test_multiple_fusions(self):
        if False:
            while True:
                i = 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 2, 3))])
        def prog(x):
            if False:
                i = 10
                return i + 15
            x = mb.transpose(x=x, perm=[0, 2, 1])
            x1 = mb.relu(x=x)
            x2 = mb.relu(x=x)
            y1 = mb.transpose(x=x1, perm=[0, 2, 1])
            x3 = mb.log(x=x2)
            y2 = mb.transpose(x=x3, perm=[0, 2, 1])
            return (y1, y2)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'relu', 'transpose', 'log', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'log'])
        assert prev_block.inputs['x'] == prev_block.find_ops(op_type='transpose')[0].inputs['x']
        assert block.find_ops(op_type='log')[0].outputs[0] in block.outputs
        assert_model_is_valid(prog, {'x': (10, 2, 3)}, expected_output_shapes={block.outputs[0].name: (10, 2, 3), block.outputs[1].name: (10, 2, 3)})
    '\n    Input graph:\n    input(shape=10,2,3,5)--->transpose(axis=[0,2,3,1])----->relu---->pool----->out1\n                                                       |\n                                                       |\n                                                       --->relu----->log---->transpose(axis=[0,3,1,2])---->out2\n\n\n    Output graph:\n    input(shape=10,2,3,5)----->relu---->transpose(axis=[0,2,3,1])---->pool----->out1\n                           |\n                           |\n                           --->relu----->log---->out2\n    '

    def test_partial_fusion_0(self):
        if False:
            while True:
                i = 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 2, 3, 5))])
        def prog(x):
            if False:
                return 10
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x1 = mb.relu(x=x)
            x2 = mb.relu(x=x)
            y1 = mb.avg_pool(x=x1, kernel_sizes=[1, 1], strides=[1, 1], pad_type='valid')
            x3 = mb.log(x=x2)
            y2 = mb.transpose(x=x3, perm=[0, 3, 1, 2])
            return (y1, y2)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'relu', 'avg_pool', 'log', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'transpose', 'avg_pool', 'log'])
        assert prev_block.inputs['x'] == prev_block.find_ops(op_type='transpose')[0].inputs['x']
        assert block.find_ops(op_type='log')[0].outputs[0] == block.outputs[1]
        assert block.find_ops(op_type='transpose')[0].outputs[0] == block.find_ops(op_type='avg_pool')[0].inputs['x']
        assert list(block.find_ops(op_type='transpose')[0].perm.val) == [0, 2, 3, 1]
        assert_model_is_valid(prog, {'x': (10, 2, 3, 5)}, expected_output_shapes={block.outputs[0].name: (10, 3, 5, 2), block.outputs[1].name: (10, 2, 3, 5)})
    '\n    Input graph:\n    input(shape=10,2,3,5)--->transpose(axis=[0,2,1,3])----->relu---->transpose(axis=[0,2,1,3])---->out1\n                                                        |\n                                                        |\n                                                        --->pool--->log---->transpose(axis=[0,2,1,3])---->out2\n\n    Output graph:\n    input(shape=10,2,3,5)----->relu---->out1\n                           |\n                           |\n                           --->transpose(axis=[0,2,1,3])---->pool----->log---->transpose(axis=[0,2,1,3])---->out2\n    '

    def test_partial_fusion_1(self):
        if False:
            for i in range(10):
                print('nop')

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 2, 3, 5))])
        def prog(x):
            if False:
                print('Hello World!')
            x = mb.transpose(x=x, perm=[0, 2, 1, 3])
            x1 = mb.relu(x=x)
            x2 = mb.avg_pool(x=x, kernel_sizes=[1, 1], strides=[1, 1], pad_type='valid')
            y1 = mb.transpose(x=x1, perm=[0, 2, 1, 3])
            x3 = mb.log(x=x2)
            y2 = mb.transpose(x=x3, perm=[0, 2, 1, 3])
            return (y1, y2)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'avg_pool', 'transpose', 'log', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'transpose', 'avg_pool', 'log', 'transpose'])
        assert block.inputs['x'] == block.find_ops(op_type='relu')[0].inputs['x']
        assert block.outputs[0] == block.find_ops(op_type='relu')[0].outputs[0]
        assert_model_is_valid(prog, {'x': (10, 2, 3, 5)}, expected_output_shapes={block.outputs[0].name: (10, 2, 3, 5), block.outputs[1].name: (10, 2, 3, 5)})
    '\n    Input graph:\n\n                                                             |-------> transpose(axis=[0,2,1,3]) ---->out1(shape=10,2,3,5)\n                                                             |\n    input(shape=10,2,3,5)-->relu-->transpose(axis=[0,2,1,3])--->relu--->transpose(axis=[0,2,1,3]) ---->out2(shape=10,2,3,5)\n                                                                     |\n                                                                     |----->pool--------------->out3(shape=10,3,2,5)\n                                                                     |\n                                                                     |----->pool--------------->out4(shape=10.3.2.5)\n\n\n    Output graph:\n\n                        |---->out1(shape=10,2,3,5)\n                        |\n    input---->relu---------->relu------->out2(shape=10,2,3,5)\n                                    |\n                                    |----->transpose(axis=[0,2,1,3])--->pool---->out3(shape=10,3,2,5)\n                                    |\n                                    |----->transpose(axis=[0,2,1,3])---->pool--->out4(shape=10.3.2.5)\n    '

    def test_partial_fusion_2(self):
        if False:
            print('Hello World!')

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 2, 3, 5))])
        def prog(x):
            if False:
                return 10
            x = mb.relu(x=x)
            x = mb.transpose(x=x, perm=[0, 2, 1, 3])
            y1 = mb.transpose(x=x, perm=[0, 2, 1, 3])
            x1 = mb.relu(x=x)
            y2 = mb.transpose(x=x1, perm=[0, 2, 1, 3])
            y3 = mb.avg_pool(x=x1, kernel_sizes=[1, 1], strides=[1, 1], pad_type='valid')
            y4 = mb.avg_pool(x=x1, kernel_sizes=[1, 1], strides=[1, 1], pad_type='valid')
            return (y1, y2, y3, y4)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['relu', 'transpose', 'transpose', 'relu', 'transpose', 'avg_pool', 'avg_pool'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'transpose', 'avg_pool', 'transpose', 'avg_pool'])
        assert block.outputs[0] == block.find_ops(op_type='relu')[0].outputs[0]
        assert block.outputs[1] == block.find_ops(op_type='relu')[1].outputs[0]
        assert_model_is_valid(prog, {'x': (10, 2, 3, 5)}, expected_output_shapes={block.outputs[0].name: (10, 2, 3, 5), block.outputs[1].name: (10, 2, 3, 5), block.outputs[2].name: (10, 3, 2, 5), block.outputs[3].name: (10, 3, 2, 5)})
    '\n    Input graph:\n\n    input(shape=10,2,3,5)-->relu--->transpose(axis=[0,2,1,3])----->transpose(axis=[0,2,1,3])---->out1(shape=10,2,3,5)\n                                                               |\n                                                               ---->relu------>out2(shape=10,3,2,5)\n\n    Output graph:\n\n    input(shape=10,2,3,5)-->relu---->out1(shape=10,2,3,5)\n                                  |\n                                   ---->relu--->transpose(axis=[0,2,1,3])------>out2(shape=10,3,2,5)\n    '

    def test_partial_fusion_3(self):
        if False:
            i = 10
            return i + 15

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 2, 3, 5))])
        def prog(x):
            if False:
                return 10
            x = mb.relu(x=x)
            x = mb.transpose(x=x, perm=[0, 2, 1, 3])
            x1 = mb.transpose(x=x, perm=[0, 2, 1, 3])
            x2 = mb.relu(x=x)
            return (x1, x2)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['relu', 'transpose', 'transpose', 'relu'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'transpose'])
        assert block.outputs[0] == block.find_ops(op_type='relu')[0].outputs[0]
        assert_model_is_valid(prog, {'x': (10, 2, 3, 5)}, expected_output_shapes={block.outputs[0].name: (10, 2, 3, 5), block.outputs[1].name: (10, 3, 2, 5)})
    '\n    Input graph:\n\n    input(shape=10,2,3,5)-->relu--->transpose(axis=[0,2,1,3])----->transpose(axis=[0,2,1,3])---->out1(shape=10,2,3,5)\n                                                               |\n                                                               ------>out2(shape=10,3,2,5)\n\n    Output graph:\n    same as input graph as one of the optimizing transpose is connected to model output\n    '

    def test_partial_fusion_4(self):
        if False:
            i = 10
            return i + 15

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 2, 3, 5))])
        def prog(x):
            if False:
                return 10
            x = mb.relu(x=x)
            out2 = mb.transpose(x=x, perm=[0, 2, 1, 3])
            out1 = mb.transpose(x=out2, perm=[0, 2, 1, 3])
            return (out1, out2)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['relu', 'transpose', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'transpose', 'transpose'])
        assert block.outputs[1] == block.find_ops(op_type='transpose')[0].outputs[0]
        assert_model_is_valid(prog, {'x': (10, 2, 3, 5)}, expected_output_shapes={block.outputs[0].name: (10, 2, 3, 5), block.outputs[1].name: (10, 3, 2, 5)})
    '\n    Input graph:\n    input(shape=10,2,3,5)-->relu-->transpose(axis=[0,2,1,3])--->relu--->transpose(axis=[0,2,1,3]) ---->out1(shape=10,2,3,5)\n                                                                     |\n                                                                     |--->relu-->pool--------------->out2(shape=10,3,2,5)\n                                                                     |\n                                                                     |----->pool--------------->out3(shape=10.3.2.5)\n\n\n    Output graph:\n    same as the input graph as materialization ops are greater than cancel ops\n    '

    def test_no_fusion_more_materialization_ops(self):
        if False:
            while True:
                i = 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 2, 3, 5))])
        def prog(x):
            if False:
                print('Hello World!')
            x = mb.relu(x=x)
            x = mb.transpose(x=x, perm=[0, 2, 1, 3])
            x1 = mb.relu(x=x)
            y2 = mb.transpose(x=x1, perm=[0, 2, 1, 3])
            x2 = mb.relu(x=x1)
            y3 = mb.avg_pool(x=x2, kernel_sizes=[1, 1], strides=[1, 1], pad_type='valid')
            y4 = mb.avg_pool(x=x1, kernel_sizes=[1, 1], strides=[1, 1], pad_type='valid')
            return (y2, y3, y4)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['relu', 'transpose', 'relu', 'transpose', 'relu', 'avg_pool', 'avg_pool'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'transpose', 'relu', 'transpose', 'relu', 'avg_pool', 'avg_pool'])
        assert_model_is_valid(prog, {'x': (10, 2, 3, 5)}, expected_output_shapes={block.outputs[0].name: (10, 2, 3, 5), block.outputs[1].name: (10, 3, 2, 5), block.outputs[2].name: (10, 3, 2, 5)})
    '\n    Input graph:\n    input(shape=10,2,3)--->transpose(axis=[0,2,1])----->relu---->transpose(axis=[0,2,1])---->out1\n                                                    |\n                                                    |\n                                                    --->reduce(axis=2)----->log---->transpose(axis=[0,2,1])---->out2\n\n    Output graph:\n    input(shape=10,2,3)----->relu---->out1\n                        |\n                        |\n                        --->reduce(axis=1)----->log---->out2\n    '

    def test_fusion_with_axis_op(self):
        if False:
            return 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 2, 3))])
        def prog(x):
            if False:
                for i in range(10):
                    print('nop')
            x = mb.transpose(x=x, perm=[0, 2, 1])
            x1 = mb.relu(x=x)
            x2 = mb.reduce_mean(x=x, axes=[2], keep_dims=True)
            y1 = mb.transpose(x=x1, perm=[0, 2, 1])
            x3 = mb.log(x=x2)
            y2 = mb.transpose(x=x3, perm=[0, 2, 1])
            return (y1, y2)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'reduce_mean', 'transpose', 'log', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'reduce_mean', 'log'])
        assert list(block.find_ops(op_type='reduce_mean')[0].inputs['axes'].val) == [1]
        assert_model_is_valid(prog, {'x': (10, 2, 3)}, expected_output_shapes={block.outputs[0].name: (10, 2, 3), block.outputs[1].name: (10, 1, 3)})
    '\n    Input graph:\n    input(shape=11,2,3,6)--->transpose(axis=[0,3,1,2])---\n                                                       |\n                                                       |\n                                                        --->pad(pad=[0,0,0,0,1,2,3,4])\n                                                              |\n                                                              |-->log--->transpose(axis=[0,2,3,1])-->out1(shape=11,5,10,6)\n\n    Output graph:\n    same as input graph, as transpose cannot be pushed through the pad op since "reflect" mode is only supported\n    along the last two axis\n    '

    def test_fusion_with_pad_reflective_op_0(self):
        if False:
            return 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(11, 2, 3, 6))])
        def prog(x):
            if False:
                while True:
                    i = 10
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x2 = mb.pad(x=x, pad=[0, 0, 0, 0, 1, 2, 3, 4], mode='reflect')
            x3 = mb.log(x=x2)
            y2 = mb.transpose(x=x3, perm=[0, 2, 3, 1])
            return y2
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'pad', 'log', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['transpose', 'pad', 'log', 'transpose'])
        assert list(block.find_ops(op_type='pad')[0].inputs['pad'].val.flatten()) == [0, 0, 0, 0, 1, 2, 3, 4]
        assert_model_is_valid(prog, {'x': (11, 2, 3, 6)}, expected_output_shapes={block.outputs[0].name: (11, 5, 10, 6)})
    '\n    Input graph:\n    input(shape=11,2,3,6)--->transpose(axis=[0,1,3,2])---\n                                                       |\n                                                       |\n                                                        --->pad(pad=[0,0,0,0,1,2,3,4])\n                                                              |\n                                                              |-->log--->transpose(axis=[0,1,3,2])-->out1(shape=11,2,10,9)\n\n    Output graph:\n    input(shape=11,2,3,6)--->pad(pad=[0,0,0,0,3,4,1,2])-->log-->out1(shape=11,2,10,9)\n    '

    def test_fusion_with_pad_reflective_op_1(self):
        if False:
            return 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(11, 2, 3, 6))])
        def prog(x):
            if False:
                while True:
                    i = 10
            x = mb.transpose(x=x, perm=[0, 1, 3, 2])
            x2 = mb.pad(x=x, pad=[0, 0, 0, 0, 1, 2, 3, 4], mode='reflect')
            x3 = mb.log(x=x2)
            y2 = mb.transpose(x=x3, perm=[0, 1, 3, 2])
            return y2
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'pad', 'log', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['pad', 'log'])
        assert list(block.find_ops(op_type='pad')[0].inputs['pad'].val.flatten()) == [0, 0, 0, 0, 3, 4, 1, 2]
        assert_model_is_valid(prog, {'x': (11, 2, 3, 6)}, expected_output_shapes={block.outputs[0].name: (11, 2, 10, 9)})
    '\n    Input graph:\n    input(shape=11,2,3,6)--->transpose(axis=[0,3,1,2])---\n                                                       |\n                                                       |\n                                                        --->pad(pad=[0,0,0,0,1,2,3,4])\n                                                              |\n                                                              |-->log--->transpose(axis=[0,2,3,1])-->out1(shape=11,5,10,6)\n\n    Output graph:\n    input(shape=11,2,3,6)--->pad(pad=[0,0,1,2,3,4,0,0])-->log-->out1(shape=11,5,10,6)\n    '

    def test_fusion_with_pad_constant_op(self):
        if False:
            print('Hello World!')

        @mb.program(input_specs=[mb.TensorSpec(shape=(11, 2, 3, 6))])
        def prog(x):
            if False:
                for i in range(10):
                    print('nop')
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x2 = mb.pad(x=x, pad=[0, 0, 0, 0, 1, 2, 3, 4], mode='constant', constant_val=3.0)
            x3 = mb.log(x=x2)
            y2 = mb.transpose(x=x3, perm=[0, 2, 3, 1])
            return y2
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'pad', 'log', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['pad', 'log'])
        assert list(block.find_ops(op_type='pad')[0].inputs['pad'].val.flatten()) == [0, 0, 1, 2, 3, 4, 0, 0]
        assert_model_is_valid(prog, {'x': (11, 2, 3, 6)}, expected_output_shapes={block.outputs[0].name: (11, 5, 10, 6)})
    '\n    Input graph:\n                                                    const(shape=2)\n                                                          |\n                                                          V\n    input(shape=1,2,5,5)--->transpose(axis=[0,2,3,1])--->add---->transpose(axis=[0,3,1,2])--->out(shape=1,2,5,5)\n\n    Output graph:\n                        const(shape=1,2,1,1)\n                             |\n                             V\n    input(shape=1,2,5,5)--->add--->out(shape=1,2,5,5)\n    '

    def test_fusion_with_add_constant_op(self):
        if False:
            return 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 5, 5))])
        def prog(x):
            if False:
                while True:
                    i = 10
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x = mb.add(x=x, y=np.array([10, 100]))
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            return x
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'add', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['add'])
        assert_model_is_valid(prog, {'x': (1, 2, 5, 5)}, expected_output_shapes={block.outputs[0].name: (1, 2, 5, 5)})
    '\n    Input graph:\n                                                    const(scalar)\n                                                          |\n                                                          V\n    input(shape=1,2,5,5)--->transpose(axis=[0,2,3,1])--->add---->transpose(axis=[0,3,1,2])--->out(shape=1,2,5,5)\n\n    Output graph:\n                        const(scalar)\n                             |\n                             V\n    input(shape=1,2,5,5)--->add--->out(shape=1,2,5,5)\n    '

    def test_fusion_with_add_scalar_constant_op(self):
        if False:
            while True:
                i = 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 5, 5))])
        def prog(x):
            if False:
                print('Hello World!')
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x = mb.add(x=5, y=x)
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            return x
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'add', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['add'])
        assert_model_is_valid(prog, {'x': (1, 2, 5, 5)}, expected_output_shapes={block.outputs[0].name: (1, 2, 5, 5)})
    '\n    Input graph:\n    input(shape=1,2,5,5)----->transpose(axis=[0,2,3,1])--->add---->transpose(axis=[0,3,1,2])--->out(shape=1,2,5,5)\n                          |                                 ^\n                          |                                 |\n                          |---->relu---->transpose(axis=[0,2,3,1])\n\n    Output graph:\n    input(shape=1,2,5,5)----->add--->out(shape=1,2,5,5)\n                      |        ^\n                      |        |\n                      |------>relu\n    '

    def test_fusion_with_add_broadcastable_0(self):
        if False:
            return 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 5, 5))])
        def prog(x):
            if False:
                i = 10
                return i + 15
            x1 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x2 = mb.relu(x=x)
            x2 = mb.transpose(x=x2, perm=[0, 2, 3, 1])
            x3 = mb.add(x=x1, y=x2)
            y = mb.transpose(x=x3, perm=[0, 3, 1, 2])
            return y
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'transpose', 'add', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'add'])
        assert block.find_ops(op_type='relu')[0].inputs['x'] == block.inputs['x']
        assert block.find_ops(op_type='add')[0].inputs['x'] == block.inputs['x']
        assert block.find_ops(op_type='add')[0].inputs['y'] == block.find_ops(op_type='relu')[0].outputs[0]
        assert_model_is_valid(prog, {'x': (1, 2, 5, 5)}, expected_output_shapes={block.outputs[0].name: (1, 2, 5, 5)})
    '\n    Input graph:\n    input(shape=1,2,5,5)----->transpose(axis=[0,2,3,1])--->add---->transpose(axis=[0,3,1,2])--->out(shape=1,2,5,5)\n                          |                                 ^\n                          |                                 |\n                          |----------------------->transpose(axis=[0,2,3,1])\n\n    Output graph:\n    input(shape=1,2,5,5)----->add--->out(shape=1,2,5,5)\n                      |        ^\n                      |        |\n                      |---------\n    '

    def test_fusion_with_add_broadcastable_1(self):
        if False:
            for i in range(10):
                print('nop')

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 5, 5))])
        def prog(x):
            if False:
                for i in range(10):
                    print('nop')
            x1 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x2 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x3 = mb.add(x=x1, y=x2)
            y = mb.transpose(x=x3, perm=[0, 3, 1, 2])
            return y
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'transpose', 'add', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['add'])
        assert block.find_ops(op_type='add')[0].inputs['x'] == block.inputs['x']
        assert block.find_ops(op_type='add')[0].inputs['y'] == block.inputs['x']
        assert_model_is_valid(prog, {'x': (1, 2, 5, 5)}, expected_output_shapes={block.outputs[0].name: (1, 2, 5, 5)})
    '\n    Input graph:\n    input(shape=1,2,5,5)--->transpose(axis=[0,2,3,1])---> relu---->concat(axis=3)----->transpose(axis=[0,3,1,2])----->out1(shape=1,4,5,5)\n                         |                                              ^\n                         |                                              |\n                         |->transpose(axis=[0,2,3,1])--->relu------------\n\n    Output graph:\n    input(shape=1,2,5,5)------> relu---->concat(axis=1)--->out1(shape=1,4,5,5)\n                         |                    ^\n                         |                    |\n                         |---->relu------------\n    '

    def test_concat_pattern_0(self):
        if False:
            print('Hello World!')

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 5, 5))])
        def prog(x):
            if False:
                while True:
                    i = 10
            x1 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x2 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x1 = mb.relu(x=x1)
            x2 = mb.relu(x=x2)
            x3 = mb.concat(values=[x1, x2], axis=3)
            x4 = mb.transpose(x=x3, perm=[0, 3, 1, 2])
            return x4
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'transpose', 'relu', 'relu', 'concat', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'concat'])
        assert_model_is_valid(prog, {'x': (1, 2, 5, 5)}, expected_output_shapes={block.outputs[0].name: (1, 4, 5, 5)})
    '\n    Input graph:\n    input(shape=1,2,5,5)--->transpose(axis=[0,2,3,1])---> relu---->concat(axis=3)----->transpose(axis=[0,3,1,2])----->out1(shape=1,4,5,5)\n                         |                                              ^\n                         |                                              |\n                         |->transpose(axis=[0,2,3,1])------->relu--------\n                                                        |\n                                                        V\n                                                       pool--->out2(shape=1,5,5,2)\n\n\n\n    Output graph:\n    input(shape=1,2,5,5)------> relu---->concat(axis=1)--->out1(shape=1,4,5,5)\n                         |                    ^\n                         |                    |\n                         |---->relu------------\n                         |\n                         |--->transpose(axis=[0,2,3,1])---->pool--->out2(shape=1,5,5,2)\n    '

    def test_concat_pattern_1(self):
        if False:
            print('Hello World!')

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 5, 5))])
        def prog(x):
            if False:
                while True:
                    i = 10
            x1 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x2 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x1 = mb.relu(x=x1)
            x2 = mb.relu(x=x2)
            x3 = mb.concat(values=[x1, x2], axis=3)
            x4 = mb.transpose(x=x3, perm=[0, 3, 1, 2])
            x5 = mb.avg_pool(x=x2, kernel_sizes=[1, 1], strides=[1, 1], pad_type='valid')
            return (x4, x5)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'transpose', 'relu', 'relu', 'concat', 'transpose', 'avg_pool'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'concat', 'transpose', 'avg_pool'])
        assert_model_is_valid(prog, {'x': (1, 2, 5, 5)}, expected_output_shapes={block.outputs[0].name: (1, 4, 5, 5), block.outputs[1].name: (1, 5, 5, 2)})
    '\n    Input graph:\n    input(shape=1,2,5,5)--->transpose(axis=[0,2,3,1])---> relu---->concat(axis=3)----->transpose(axis=[0,3,1,2])----->out1(shape=1,4,5,5)\n                         |                                              ^\n                         |                                              |\n                         |->transpose(axis=[0,2,3,1])------->relu--------\n                                                        |\n                                                        V\n                                                       relu--->out2(shape=1,5,5,2)\n\n\n\n    Output graph:\n    input(shape=1,2,5,5)------> relu---->concat(axis=1)--->out1(shape=1,4,5,5)\n                         |                    ^\n                         |                    |\n                         |---->relu------------\n                         |\n                         |--->relu---->transpose(axis=[0,2,3,1])---->out2(shape=1,5,5,2)\n    '

    def test_concat_pattern_2(self):
        if False:
            i = 10
            return i + 15

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 5, 5))])
        def prog(x):
            if False:
                print('Hello World!')
            x1 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x2 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x1 = mb.relu(x=x1)
            x2 = mb.relu(x=x2)
            x3 = mb.concat(values=[x1, x2], axis=3)
            x4 = mb.transpose(x=x3, perm=[0, 3, 1, 2])
            x5 = mb.relu(x=x2)
            return (x4, x5)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'transpose', 'relu', 'relu', 'concat', 'transpose', 'relu'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'concat', 'relu', 'transpose'])
        assert_model_is_valid(prog, {'x': (1, 2, 5, 5)}, expected_output_shapes={block.outputs[0].name: (1, 4, 5, 5), block.outputs[1].name: (1, 5, 5, 2)})
    '\n    Input graph:\n    input(shape=1,2,5,5)--->transpose(axis=[0,2,3,1])---> relu---->concat(axis=3)----->transpose(axis=[0,3,1,2])----->out1(shape=1,4,5,5)\n                         |                                              ^\n                         |                                              |\n                         |->transpose(axis=[0,2,3,1])------->relu--------\n                                                        |\n                                                        V\n                                                       out2(shape=1,5,5,2)\n\n\n\n    Output graph:\n    input(shape=1,2,5,5)------> relu---->concat(axis=1)--->out1(shape=1,4,5,5)\n                         |                    ^\n                         |                    |\n                         |---->relu------------\n                         |\n                         |--->transpose(axis=[0,2,3,1])---->out2(shape=1,5,5,2)\n    '

    def test_concat_pattern_3(self):
        if False:
            while True:
                i = 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 5, 5))])
        def prog(x):
            if False:
                for i in range(10):
                    print('nop')
            x1 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x2 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x1 = mb.relu(x=x1)
            x2 = mb.relu(x=x2)
            x3 = mb.concat(values=[x1, x2], axis=3)
            x4 = mb.transpose(x=x3, perm=[0, 3, 1, 2])
            return (x4, x2)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'transpose', 'relu', 'relu', 'concat', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'concat', 'transpose'])
        assert_model_is_valid(prog, {'x': (1, 2, 5, 5)}, expected_output_shapes={block.outputs[0].name: (1, 4, 5, 5), block.outputs[1].name: (1, 5, 5, 2)})
    '\n    Input graph:\n    input(shape=1,2,5,5)--->transpose(axis=[0,2,3,1])---> relu---->concat(axis=3)----->transpose(axis=[0,3,1,2])----->out1(shape=1,4,5,5)\n                         |                                              ^\n                         |                                              |\n                         |->transpose(axis=[0,2,3,1])------->relu--------\n                                                        |\n                                                        V\n                                            transpose(axis=[0,3,1,2]) -----> out2(shape=1,2,5,5)\n\n    Output graph:\n    input(shape=1,2,5,5)---> relu---->concat(axis=1)----->out1(shape=1,4,5,5)\n                     |                     ^\n                     |                     |\n                     |------------------->relu-------->out2(shape=1,2,5,5)\n    '

    def test_concat_pattern_4(self):
        if False:
            print('Hello World!')

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 5, 5))])
        def prog(x):
            if False:
                print('Hello World!')
            x1 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x2 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x1 = mb.relu(x=x1)
            x2 = mb.relu(x=x2)
            x3 = mb.concat(values=[x1, x2], axis=3)
            x4 = mb.transpose(x=x3, perm=[0, 3, 1, 2])
            x5 = mb.transpose(x=x2, perm=[0, 3, 1, 2])
            return (x4, x5)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'transpose', 'relu', 'relu', 'concat', 'transpose', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'concat'])
        assert_model_is_valid(prog, {'x': (1, 2, 5, 5)}, expected_output_shapes={block.outputs[0].name: (1, 4, 5, 5), block.outputs[1].name: (1, 2, 5, 5)})
    '\n    Input graph:\n                                                     constant(shape=[30,10,5])\n                                                            |\n                                                            V\n    input(shape=10,20,30)--->transpose(axis=[2,0,1])--->concat(axis=2)----->transpose(axis=[1,2,0])----->out1(shape=10,25,30)\n\n    Output graph:\n                        constant(shape=[10,5,30])\n                                |\n                                V\n    input(shape=10,20,30)--->concat(axis=1)----->out1(shape=10,25,30)\n    '

    def test_concat_pattern_5(self):
        if False:
            for i in range(10):
                print('nop')
        const = np.random.rand(30, 10, 5)

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20, 30))])
        def prog(x):
            if False:
                print('Hello World!')
            x1 = mb.transpose(x=x, perm=[2, 0, 1])
            c = mb.const(val=const)
            x2 = mb.concat(values=[x1, c], axis=2)
            x3 = mb.transpose(x=x2, perm=[1, 2, 0])
            return x3
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'concat', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['concat'])
        assert_model_is_valid(prog, {'x': (10, 20, 30)}, expected_output_shapes={block.outputs[0].name: (10, 25, 30)})
    '\n    Input graph:\n                                        input2(shape=30,10,20)-----|\n                                                                   |\n    input(shape=10,20,30)--->transpose(axis=[2,0,1])----->relu-----|----->concat(axis=2)------>out1(shape=90,10,20)\n                                                      |            |\n                                                      |-->relu-----|\n                                                      |\n                                                      |-->relu---->transpose(axis=[1,2,0])---->out2(shape=10,20,30)\n                                                      |\n                                                      |-->relu---->transpose(axis=[1,2,0])---->out3(shape=10,20,30)\n                                                      |\n                                                      |-->relu---->transpose(axis=[1,2,0])---->out4(shape=10,20,30)\n\n    Output graph:\n\n                                        input2(shape=30,10,20)-----|\n                                                                   |\n    input(shape=10,20,30)----->relu--->transpose(axis=[2,0,1])-----|----->concat(axis=2)------>out1(shape=90,10,20)\n                           |                                       |\n                           |-->relu--->transpose(axis=[2,0,1])-----|\n                           |\n                           |-->relu---->out2(shape=10,20,30)\n                           |\n                           |-->relu---->out3(shape=10,20,30)\n                           |\n                           |-->relu---->out4(shape=10,20,30)\n\n    Output graph:\n    '

    def test_concat_pattern_6(self):
        if False:
            while True:
                i = 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20, 30)), mb.TensorSpec(shape=(30, 10, 20))])
        def prog(x, y):
            if False:
                i = 10
                return i + 15
            x1 = mb.transpose(x=x, perm=[2, 0, 1])
            r1 = mb.relu(x=x1)
            r2 = mb.relu(x=x1)
            r3 = mb.relu(x=x1)
            r4 = mb.relu(x=x1)
            r5 = mb.relu(x=x1)
            x2 = mb.concat(values=[r1, r2, y], axis=0)
            x3 = mb.transpose(x=r3, perm=[1, 2, 0])
            x4 = mb.transpose(x=r4, perm=[1, 2, 0])
            x5 = mb.transpose(x=r5, perm=[1, 2, 0])
            return (x2, x3, x4, x5)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'relu', 'relu', 'relu', 'relu', 'concat', 'transpose', 'transpose', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'relu', 'relu', 'relu', 'transpose', 'transpose', 'concat'])
        assert_model_is_valid(prog, {'x': (10, 20, 30), 'y': (30, 10, 20)}, expected_output_shapes={block.outputs[0].name: (90, 10, 20), block.outputs[1].name: (10, 20, 30), block.outputs[2].name: (10, 20, 30), block.outputs[3].name: (10, 20, 30)})
    '\n    Input graph:\n    input(shape=1,5,5,3)----->transpose(axis=[0,3,1,2])\n                                |\n                                ---->relu-------------->transpose(axis=[0,2,3,1])\n                                             |                  |\n                                             |                  V\n                                             |                 relu\n                                             |                  |\n                                             |                  V\n                                             |         transpose(axis=[0,3,1,2])\n                                             |                  |\n                                             |                  V\n                                             ----------------> add --------> relu---->pool---->out(shape=1,3,5,5)\n\n\n    Output graph:\n\n\n    input(shape=1,5,5,3)---->relu------------------------> relu\n                                         |                  |\n                                         |                  V\n                                         ----------------> add\n                                                            |\n                                                            V\n                                                           relu\n                                                            |\n                                                            V\n                                                           transpose(axis=[0,3,1,2])-->pool---->out(shape=1,3,5,5)\n\n    '

    def test_skip_connection_pattern_0(self):
        if False:
            return 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 5, 3))])
        def prog(x):
            if False:
                i = 10
                return i + 15
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x = mb.relu(x=x)
            x1 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x2 = mb.relu(x=x1)
            x3 = mb.transpose(x=x2, perm=[0, 3, 1, 2])
            x4 = mb.add(x=x, y=x3)
            x5 = mb.relu(x=x4)
            x6 = mb.avg_pool(x=x5, kernel_sizes=[1, 1], strides=[1, 1], pad_type='valid')
            return x6
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'transpose', 'relu', 'transpose', 'add', 'relu', 'avg_pool'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'add', 'relu', 'transpose', 'avg_pool'])
        assert_model_is_valid(prog, {'x': (1, 5, 5, 3)}, expected_output_shapes={block.outputs[0].name: (1, 3, 5, 5)})
    '\n    Input graph:\n    input(shape=1,5,5,3)----->transpose(axis=[0,3,1,2])\n                                |\n                                ---->relu-------------->transpose(axis=[0,2,3,1])\n                                             |                  |\n                                             |                  V\n                                             |                 relu\n                                             |                  |\n                                             |                  V\n                                             |         transpose(axis=[0,3,1,2])\n                                             |                  |\n                                             |                  V\n                                             ----------------> add -->transpose(axis=[0,2,3,1])\n                                                                            |\n                                                                            V\n                                                                          relu---->pool---->out(shape=1,5,5,3)\n\n\n    Output graph:\n\n\n    input(shape=1,5,5,3)---->relu------------------------> relu\n                                         |                  |\n                                         |                  V\n                                         ----------------> add\n                                                            |\n                                                            V\n                                                           relu\n                                                            |\n                                                            V\n                                                           pool---->out(shape=1,5,5,3)\n\n    '

    def test_skip_connection_pattern_1(self):
        if False:
            for i in range(10):
                print('nop')

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 5, 3))])
        def prog(x):
            if False:
                for i in range(10):
                    print('nop')
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x = mb.relu(x=x)
            x1 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x2 = mb.relu(x=x1)
            x3 = mb.transpose(x=x2, perm=[0, 3, 1, 2])
            x4 = mb.add(x=x, y=x3)
            x4 = mb.transpose(x=x4, perm=[0, 2, 3, 1])
            x5 = mb.relu(x=x4)
            x6 = mb.avg_pool(x=x5, kernel_sizes=[1, 1], strides=[1, 1], pad_type='valid')
            return x6
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'transpose', 'relu', 'transpose', 'add', 'transpose', 'relu', 'avg_pool'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'relu', 'add', 'relu', 'avg_pool'])
        assert_model_is_valid(prog, {'x': (1, 5, 5, 3)}, expected_output_shapes={block.outputs[0].name: (1, 5, 5, 3)})
    '\n    Input graph:\n    input(shape=2,5)--->transpose(axis=[1,0])--->transpose(axis=[1,0])-->reduce(axis=1)\n                                |                                             |\n                                |                                             V\n                                |                                    transpose(axis=[1,0])\n                                |                                             |\n                                |                                             V\n                                -------------------------------------------->add------->out(shape=5,2)\n\n    Output graph:\n    input(shape=2,5)--->reduce(axis=1)---->add---->transpose(axis=[1,0])--->out(shape=5,2)\n                     |                      ^\n                     |                      |\n                     ------------------------\n    '

    def test_residual_with_unmaterialized_output(self):
        if False:
            return 10

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 5))])
        def prog(x):
            if False:
                for i in range(10):
                    print('nop')
            x1 = mb.transpose(x=x, perm=[1, 0])
            t1 = mb.transpose(x=x1, perm=[1, 0])
            x2 = mb.reduce_mean(x=t1, axes=[1], keep_dims=True)
            t2 = mb.transpose(x=x2, perm=[1, 0])
            return mb.add(x=x1, y=t2)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'transpose', 'reduce_mean', 'transpose', 'add'])
        self.assertEqual(get_op_types_in_program(prog), ['reduce_mean', 'add', 'transpose'])
        assert_model_is_valid(prog, {'x': (2, 5)}, expected_output_shapes={block.outputs[0].name: (5, 2)})
    '\n    Input graph:\n    input(shape=2,5)--->transpose(axis=[1,0])--->transpose(axis=[1,0])-->reduce(axis=1)\n                                |                                             |\n                                |                                             V\n                                |                                    transpose(axis=[1,0])\n                                |                                             |\n                                |                                             V\n                                -------------------------------------------->add------->out1(shape=5,2)\n                                                                              |\n                                                                              V\n                                                                            relu------->out2(shape=5,2)\n\n    Output graph:\n    input(shape=2,5)--->reduce(axis=1)----> add ----->transpose(axis=[1,0])----->out1(shape=5,2)\n                     |                       |\n                     |                       V\n                     ---------------------> relu----->transpose(axis=[1,0])----->out2(shape=5,2)\n    '

    def test_residual_with_unmaterialized_multiple_output(self):
        if False:
            i = 10
            return i + 15

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 5))])
        def prog(x):
            if False:
                i = 10
                return i + 15
            x1 = mb.transpose(x=x, perm=[1, 0])
            t1 = mb.transpose(x=x1, perm=[1, 0])
            x2 = mb.reduce_mean(x=t1, axes=[1], keep_dims=True)
            t2 = mb.transpose(x=x2, perm=[1, 0])
            out1 = mb.add(x=x1, y=t2)
            out2 = mb.relu(x=out1)
            return (out1, out2)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'transpose', 'reduce_mean', 'transpose', 'add', 'relu'])
        self.assertEqual(get_op_types_in_program(prog), ['reduce_mean', 'add', 'relu', 'transpose', 'transpose'])
        assert_model_is_valid(prog, {'x': (2, 5)}, expected_output_shapes={block.outputs[0].name: (5, 2), block.outputs[1].name: (5, 2)})
    '\n    Input graph:\n    input(shape=2,5)---->transpose(axis=[1,0])------>relu----->transpose(axis=[1,0])------>out2(shape=2,5)\n                                                       |\n                                                       ------->out1(shape=5,2)\n\n    Output graph:\n    input(shape=2,5)---->relu-----> out2(shape=2,5)\n                           |\n                           V\n                    transpose(axis=[1,0]) -----> out1(shape=5,2)\n    '

    def test_materialized_output_reuse(self):
        if False:
            print('Hello World!')

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 5))])
        def prog(x):
            if False:
                print('Hello World!')
            x1 = mb.transpose(x=x, perm=[1, 0])
            y1 = mb.relu(x=x1)
            y2 = mb.transpose(x=y1, perm=[1, 0])
            return (y1, y2)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'transpose'])
        assert_model_is_valid(prog, {'x': (2, 5)}, expected_output_shapes={block.outputs[0].name: (5, 2), block.outputs[1].name: (2, 5)})
    '\n    Input graph:\n    input(shape=1,2,5,5)----->transpose(axis=[0,2,3,1])------->add------------>transpose(axis=[0,3,1,2])--->out1(shape=1,2,5,5)\n                                                          |     ^        |\n                                                          |     |        |\n                                                          ---->relu      ----->transpose(axis=[0,3,1,2])--->out2(shape=1,2,5,5)\n\n    Output graph:\n    input(shape=1,2,5,5)----->add------->out1(shape=1,2,5,5)\n                      |        ^    |\n                      |        |    |\n                      |------>relu  ------identity(renaming)---->out2(shape=1,2,5,5)\n    '

    def test_fusion_with_double_outputs(self):
        if False:
            i = 10
            return i + 15

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 5, 5))])
        def prog(x):
            if False:
                print('Hello World!')
            x1 = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x2 = mb.relu(x=x1)
            x3 = mb.add(x=x1, y=x2)
            y1 = mb.transpose(x=x3, perm=[0, 3, 1, 2])
            y2 = mb.transpose(x=x3, perm=[0, 3, 1, 2])
            return (y1, y2)
        (prev_prog, prev_block, block) = apply_pass_and_basic_check(prog, 'common::reduce_transposes')
        self.assertEqual(get_op_types_in_program(prev_prog), ['transpose', 'relu', 'add', 'transpose', 'transpose'])
        self.assertEqual(get_op_types_in_program(prog), ['relu', 'add', 'identity'])
        assert block.find_ops(op_type='relu')[0].inputs['x'] == block.inputs['x']
        assert block.find_ops(op_type='add')[0].inputs['x'] == block.inputs['x']
        assert block.find_ops(op_type='add')[0].inputs['y'] == block.find_ops(op_type='relu')[0].outputs[0]
        assert_model_is_valid(prog, {'x': (1, 2, 5, 5)}, expected_output_shapes={block.outputs[0].name: (1, 2, 5, 5)})