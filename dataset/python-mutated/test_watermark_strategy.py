from pyflink.common import Duration
from pyflink.common.watermark_strategy import WatermarkStrategy
from pyflink.java_gateway import get_gateway
from pyflink.util.java_utils import is_instance_of, get_field_value
from pyflink.testing.test_case_utils import PyFlinkTestCase

class WatermarkStrategyTests(PyFlinkTestCase):

    def test_with_idleness(self):
        if False:
            return 10
        jvm = get_gateway().jvm
        j_watermark_strategy = WatermarkStrategy.no_watermarks().with_idleness(Duration.of_seconds(5))._j_watermark_strategy
        self.assertTrue(is_instance_of(j_watermark_strategy, jvm.org.apache.flink.api.common.eventtime.WatermarkStrategyWithIdleness))
        self.assertEqual(get_field_value(j_watermark_strategy, 'idlenessTimeout').toMillis(), 5000)

    def test_with_watermark_alignment(self):
        if False:
            return 10
        jvm = get_gateway().jvm
        j_watermark_strategy = WatermarkStrategy.no_watermarks().with_watermark_alignment('alignment-group-1', Duration.of_seconds(20), Duration.of_seconds(10))._j_watermark_strategy
        self.assertTrue(is_instance_of(j_watermark_strategy, jvm.org.apache.flink.api.common.eventtime.WatermarksWithWatermarkAlignment))
        alignment_parameters = j_watermark_strategy.getAlignmentParameters()
        self.assertEqual(alignment_parameters.getWatermarkGroup(), 'alignment-group-1')
        self.assertEqual(alignment_parameters.getMaxAllowedWatermarkDrift(), 20000)
        self.assertEqual(alignment_parameters.getUpdateInterval(), 10000)

    def test_for_monotonous_timestamps(self):
        if False:
            return 10
        jvm = get_gateway().jvm
        j_watermark_strategy = WatermarkStrategy.for_monotonous_timestamps()._j_watermark_strategy
        self.assertTrue(is_instance_of(j_watermark_strategy.createWatermarkGenerator(None), jvm.org.apache.flink.api.common.eventtime.AscendingTimestampsWatermarks))

    def test_for_bounded_out_of_orderness(self):
        if False:
            i = 10
            return i + 15
        jvm = get_gateway().jvm
        j_watermark_strategy = WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(3))._j_watermark_strategy
        j_watermark_generator = j_watermark_strategy.createWatermarkGenerator(None)
        self.assertTrue(is_instance_of(j_watermark_generator, jvm.org.apache.flink.api.common.eventtime.BoundedOutOfOrdernessWatermarks))
        self.assertEqual(get_field_value(j_watermark_generator, 'outOfOrdernessMillis'), 3000)

    def test_no_watermarks(self):
        if False:
            return 10
        jvm = get_gateway().jvm
        j_watermark_strategy = WatermarkStrategy.no_watermarks()._j_watermark_strategy
        self.assertTrue(is_instance_of(j_watermark_strategy.createWatermarkGenerator(None), jvm.org.apache.flink.api.common.eventtime.NoWatermarksGenerator))