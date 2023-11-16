"""Module to build pipeline fragment that produces given PCollections.

For internal use only; no backwards-compatibility guarantees.
"""
import apache_beam as beam
from apache_beam.pipeline import AppliedPTransform
from apache_beam.pipeline import PipelineVisitor
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.testing.test_stream import TestStream

class PipelineFragment(object):
    """A fragment of a pipeline definition.

  A pipeline fragment is built from the original pipeline definition to include
  only PTransforms that are necessary to produce the given PCollections.
  """

    def __init__(self, pcolls, options=None):
        if False:
            i = 10
            return i + 15
        'Constructor of PipelineFragment.\n\n    Args:\n      pcolls: (List[PCollection]) a list of PCollections to build pipeline\n          fragment for.\n      options: (PipelineOptions) the pipeline options for the implicit\n          pipeline run.\n    '
        assert len(pcolls) > 0, 'Need at least 1 PCollection as the target data to build a pipeline fragment that produces it.'
        for pcoll in pcolls:
            assert isinstance(pcoll, beam.pvalue.PCollection), '{} is not an apache_beam.pvalue.PCollection.'.format(pcoll)
        self._user_pipeline = pcolls[0].pipeline
        self._pcolls = set(pcolls)
        for pcoll in self._pcolls:
            assert pcoll.pipeline is self._user_pipeline, '{} belongs to a different user pipeline than other PCollections given and cannot be used to build a pipeline fragment that produces the given PCollections.'.format(pcoll)
        self._options = options
        self._runner_pipeline = self._build_runner_pipeline()
        (_, self._context) = self._runner_pipeline.to_runner_api(return_context=True)
        from apache_beam.runners.interactive import pipeline_instrument as instr
        self._runner_pcoll_to_id = instr.pcoll_to_pcoll_id(self._runner_pipeline, self._context)
        self._id_to_target_pcoll = self._calculate_target_pcoll_ids()
        self._label_to_user_transform = self._calculate_user_transform_labels()
        (self._runner_pcolls_to_user_pcolls, self._runner_transforms_to_user_transforms) = self._build_correlation_between_pipelines(self._runner_pcoll_to_id, self._id_to_target_pcoll, self._label_to_user_transform)
        (self._necessary_transforms, self._necessary_pcollections) = self._mark_necessary_transforms_and_pcolls(self._runner_pcolls_to_user_pcolls)
        self._runner_pipeline = self._prune_runner_pipeline_to_fragment(self._runner_pipeline, self._necessary_transforms)

    def deduce_fragment(self):
        if False:
            print('Hello World!')
        'Deduce the pipeline fragment as an apache_beam.Pipeline instance.'
        fragment = beam.pipeline.Pipeline.from_runner_api(self._runner_pipeline.to_runner_api(), self._runner_pipeline.runner, self._options)
        ie.current_env().add_derived_pipeline(self._runner_pipeline, fragment)
        return fragment

    def run(self, display_pipeline_graph=False, use_cache=True, blocking=False):
        if False:
            i = 10
            return i + 15
        'Shorthand to run the pipeline fragment.'
        from apache_beam.runners.interactive.interactive_runner import InteractiveRunner
        if not isinstance(self._runner_pipeline.runner, InteractiveRunner):
            raise RuntimeError('Please specify InteractiveRunner when creating the Beam pipeline to use this function.')
        try:
            preserved_skip_display = self._runner_pipeline.runner._skip_display
            preserved_force_compute = self._runner_pipeline.runner._force_compute
            preserved_blocking = self._runner_pipeline.runner._blocking
            self._runner_pipeline.runner._skip_display = not display_pipeline_graph
            self._runner_pipeline.runner._force_compute = not use_cache
            self._runner_pipeline.runner._blocking = blocking
            return self.deduce_fragment().run()
        finally:
            self._runner_pipeline.runner._skip_display = preserved_skip_display
            self._runner_pipeline.runner._force_compute = preserved_force_compute
            self._runner_pipeline.runner._blocking = preserved_blocking

    def _build_runner_pipeline(self):
        if False:
            print('Hello World!')
        runner_pipeline = beam.pipeline.Pipeline.from_runner_api(self._user_pipeline.to_runner_api(), self._user_pipeline.runner, self._options)
        ie.current_env().add_derived_pipeline(self._user_pipeline, runner_pipeline)
        return runner_pipeline

    def _calculate_target_pcoll_ids(self):
        if False:
            while True:
                i = 10
        pcoll_id_to_target_pcoll = {}
        for pcoll in self._pcolls:
            pcoll_id_to_target_pcoll[self._runner_pcoll_to_id.get(str(pcoll), '')] = pcoll
        return pcoll_id_to_target_pcoll

    def _calculate_user_transform_labels(self):
        if False:
            i = 10
            return i + 15
        label_to_user_transform = {}

        class UserTransformVisitor(PipelineVisitor):

            def enter_composite_transform(self, transform_node):
                if False:
                    i = 10
                    return i + 15
                self.visit_transform(transform_node)

            def visit_transform(self, transform_node):
                if False:
                    print('Hello World!')
                if transform_node is not None:
                    label_to_user_transform[transform_node.full_label] = transform_node
        v = UserTransformVisitor()
        self._runner_pipeline.visit(v)
        return label_to_user_transform

    def _build_correlation_between_pipelines(self, runner_pcoll_to_id, id_to_target_pcoll, label_to_user_transform):
        if False:
            i = 10
            return i + 15
        runner_pcolls_to_user_pcolls = {}
        runner_transforms_to_user_transforms = {}

        class CorrelationVisitor(PipelineVisitor):

            def enter_composite_transform(self, transform_node):
                if False:
                    print('Hello World!')
                self.visit_transform(transform_node)

            def visit_transform(self, transform_node):
                if False:
                    while True:
                        i = 10
                self._process_transform(transform_node)
                for in_pcoll in transform_node.inputs:
                    self._process_pcoll(in_pcoll)
                for out_pcoll in transform_node.outputs.values():
                    self._process_pcoll(out_pcoll)

            def _process_pcoll(self, pcoll):
                if False:
                    return 10
                pcoll_id = runner_pcoll_to_id.get(str(pcoll), '')
                if pcoll_id in id_to_target_pcoll:
                    runner_pcolls_to_user_pcolls[pcoll] = id_to_target_pcoll[pcoll_id]

            def _process_transform(self, transform_node):
                if False:
                    for i in range(10):
                        print('nop')
                if transform_node.full_label in label_to_user_transform:
                    runner_transforms_to_user_transforms[transform_node] = label_to_user_transform[transform_node.full_label]
        v = CorrelationVisitor()
        self._runner_pipeline.visit(v)
        return (runner_pcolls_to_user_pcolls, runner_transforms_to_user_transforms)

    def _mark_necessary_transforms_and_pcolls(self, runner_pcolls_to_user_pcolls):
        if False:
            i = 10
            return i + 15
        necessary_transforms = set()
        all_inputs = set()
        updated_all_inputs = set(runner_pcolls_to_user_pcolls.keys())
        while len(updated_all_inputs) != len(all_inputs):
            all_inputs = set(updated_all_inputs)
            for pcoll in all_inputs:
                producer = pcoll.producer
                while producer:
                    if producer in necessary_transforms:
                        break
                    necessary_transforms.add(producer)
                    if producer.parent is not None:
                        necessary_transforms.update(producer.parts)
                        for part in producer.parts:
                            updated_all_inputs.update(part.outputs.values())
                    updated_all_inputs.update(producer.inputs)
                    side_input_pvalues = set(map(lambda side_input: side_input.pvalue, producer.side_inputs))
                    updated_all_inputs.update(side_input_pvalues)
                    producer = producer.parent
        return (necessary_transforms, all_inputs)

    def _prune_runner_pipeline_to_fragment(self, runner_pipeline, necessary_transforms):
        if False:
            for i in range(10):
                print('nop')

        class PruneVisitor(PipelineVisitor):

            def enter_composite_transform(self, transform_node):
                if False:
                    print('Hello World!')
                if should_skip_pruning(transform_node):
                    return
                pruned_parts = list(transform_node.parts)
                for part in transform_node.parts:
                    if part not in necessary_transforms:
                        pruned_parts.remove(part)
                transform_node.parts = tuple(pruned_parts)
                self.visit_transform(transform_node)

            def visit_transform(self, transform_node):
                if False:
                    print('Hello World!')
                if transform_node not in necessary_transforms:
                    transform_node.parent = None
        v = PruneVisitor()
        runner_pipeline.visit(v)
        return runner_pipeline

def should_skip_pruning(transform: AppliedPTransform):
    if False:
        return 10
    return isinstance(transform.transform, TestStream) or '_DataFrame_' in transform.full_label