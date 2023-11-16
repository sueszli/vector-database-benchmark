"""Module to verify implicit cache transforms applied by Interactive Beam.

For internal use only; no backwards-compatibility guarantees.
This utility should only be used by Interactive Beam tests. For example, it can
be used to verify if the implicit cache transforms are applied as expected when
running a pipeline with the InteractiveRunner. It can also be used to verify if
a pipeline fragment has pruned unnecessary transforms. It shouldn't be used to
verify equivalence between pipelines if the code to be tested depends on or
mutates user code within transforms in pipelines.
"""

def assert_pipeline_equal(test_case, expected_pipeline, actual_pipeline):
    if False:
        i = 10
        return i + 15
    'Asserts the equivalence between two given apache_beam.Pipeline instances.\n\n    Args:\n      test_case: (unittest.TestCase) the unittest testcase where it asserts.\n      expected_pipeline: (Pipeline) the pipeline instance expected.\n      actual_pipeline: (Pipeline) the actual pipeline instance to be asserted.\n    '
    expected_pipeline_proto = expected_pipeline.to_runner_api(use_fake_coders=True)
    actual_pipeline_proto = actual_pipeline.to_runner_api(use_fake_coders=True)
    assert_pipeline_proto_equal(test_case, expected_pipeline_proto, actual_pipeline_proto)

def assert_pipeline_proto_equal(test_case, expected_pipeline_proto, actual_pipeline_proto):
    if False:
        while True:
            i = 10
    'Asserts the equivalence between two pipeline proto representations.'
    components1 = expected_pipeline_proto.components
    components2 = actual_pipeline_proto.components
    test_case.assertEqual(len(components1.transforms), len(components2.transforms))
    test_case.assertEqual(len(components1.pcollections), len(components2.pcollections))
    test_case.assertLessEqual(len(components1.windowing_strategies), len(components2.windowing_strategies))
    test_case.assertLessEqual(len(components1.coders), len(components2.coders))
    _assert_transform_equal(test_case, actual_pipeline_proto, actual_pipeline_proto.root_transform_ids[0], expected_pipeline_proto, expected_pipeline_proto.root_transform_ids[0])

def assert_pipeline_proto_contain_top_level_transform(test_case, pipeline_proto, transform_label):
    if False:
        i = 10
        return i + 15
    'Asserts the top level transforms contain a transform with the given\n   transform label.'
    _assert_pipeline_proto_contains_top_level_transform(test_case, pipeline_proto, transform_label, True)

def assert_pipeline_proto_not_contain_top_level_transform(test_case, pipeline_proto, transform_label):
    if False:
        print('Hello World!')
    'Asserts the top level transforms do not contain a transform with the given\n   transform label.'
    _assert_pipeline_proto_contains_top_level_transform(test_case, pipeline_proto, transform_label, False)

def _assert_pipeline_proto_contains_top_level_transform(test_case, pipeline_proto, transform_label, contain):
    if False:
        while True:
            i = 10
    top_level_transform_labels = pipeline_proto.components.transforms[pipeline_proto.root_transform_ids[0]].subtransforms
    test_case.assertEqual(contain, any((transform_label in top_level_transform_label for top_level_transform_label in top_level_transform_labels)))

def _assert_transform_equal(test_case, expected_pipeline_proto, expected_transform_id, actual_pipeline_proto, actual_transform_id):
    if False:
        return 10
    'Asserts the equivalence between transforms from two given pipelines. '
    transform_proto1 = expected_pipeline_proto.components.transforms[expected_transform_id]
    transform_proto2 = actual_pipeline_proto.components.transforms[actual_transform_id]
    test_case.assertEqual(transform_proto1.spec.urn, transform_proto2.spec.urn)
    test_case.assertEqual(len(transform_proto1.subtransforms), len(transform_proto2.subtransforms))
    test_case.assertSetEqual(set(transform_proto1.inputs), set(transform_proto2.inputs))
    test_case.assertSetEqual(set(transform_proto1.outputs), set(transform_proto2.outputs))