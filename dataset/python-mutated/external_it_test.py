"""Integration tests for cross-language transform expansion."""
import unittest
import pytest
import apache_beam as beam
from apache_beam import Pipeline
from apache_beam.runners.portability import expansion_service
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms import ptransform

class ExternalTransformIT(unittest.TestCase):

    @pytest.mark.it_postcommit
    def test_job_python_from_python_it(self):
        if False:
            while True:
                i = 10

        @ptransform.PTransform.register_urn('simple', None)
        class SimpleTransform(ptransform.PTransform):

            def expand(self, pcoll):
                if False:
                    i = 10
                    return i + 15
                return pcoll | beam.Map(lambda x: 'Simple(%s)' % x)

            def to_runner_api_parameter(self, unused_context):
                if False:
                    print('Hello World!')
                return ('simple', None)

            @staticmethod
            def from_runner_api_parameter(_0, _1, _2):
                if False:
                    for i in range(10):
                        print('nop')
                return SimpleTransform()
        pipeline = TestPipeline(is_integration_test=True)
        res = pipeline | beam.Create(['a', 'b']) | beam.ExternalTransform('simple', None, expansion_service.ExpansionServiceServicer())
        assert_that(res, equal_to(['Simple(a)', 'Simple(b)']))
        (proto_pipeline, _) = pipeline.to_runner_api(return_context=True)
        pipeline_from_proto = Pipeline.from_runner_api(proto_pipeline, pipeline.runner, pipeline._options)
        pipeline_from_proto.run().wait_until_finish()
if __name__ == '__main__':
    unittest.main()