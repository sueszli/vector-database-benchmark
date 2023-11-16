"""ConsumerTrackingPipelineVisitor, a PipelineVisitor object."""
from typing import TYPE_CHECKING
from typing import Dict
from typing import Set
from apache_beam import pvalue
from apache_beam.pipeline import PipelineVisitor
if TYPE_CHECKING:
    from apache_beam.pipeline import AppliedPTransform

class ConsumerTrackingPipelineVisitor(PipelineVisitor):
    """For internal use only; no backwards-compatibility guarantees.

  Visitor for extracting value-consumer relations from the graph.

  Tracks the AppliedPTransforms that consume each PValue in the Pipeline. This
  is used to schedule consuming PTransforms to consume input after the upstream
  transform has produced and committed output.
  """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.value_to_consumers = {}
        self.root_transforms = set()
        self.step_names = {}
        self._num_transforms = 0
        self._views = set()

    @property
    def views(self):
        if False:
            i = 10
            return i + 15
        'Returns a list of side intputs extracted from the graph.\n\n    Returns:\n      A list of pvalue.AsSideInput.\n    '
        return list(self._views)

    def visit_transform(self, applied_ptransform):
        if False:
            for i in range(10):
                print('nop')
        inputs = list(applied_ptransform.inputs)
        if inputs:
            for input_value in inputs:
                if isinstance(input_value, pvalue.PBegin):
                    self.root_transforms.add(applied_ptransform)
                if input_value not in self.value_to_consumers:
                    self.value_to_consumers[input_value] = set()
                self.value_to_consumers[input_value].add(applied_ptransform)
        else:
            self.root_transforms.add(applied_ptransform)
        self.step_names[applied_ptransform] = 's%d' % self._num_transforms
        self._num_transforms += 1
        for side_input in applied_ptransform.side_inputs:
            self._views.add(side_input)