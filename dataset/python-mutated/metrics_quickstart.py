from random import random
import time
from opencensus.ext.stackdriver import stats_exporter
from opencensus.stats import aggregation
from opencensus.stats import measure
from opencensus.stats import stats
from opencensus.stats import view
LATENCY_MS = measure.MeasureFloat('task_latency', 'The task latency in milliseconds', 'ms')
LATENCY_VIEW = view.View('task_latency_distribution', 'The distribution of the task latencies', [], LATENCY_MS, aggregation.DistributionAggregation([100.0, 200.0, 400.0, 1000.0, 2000.0, 4000.0]))

def main():
    if False:
        return 10
    stats.stats.view_manager.register_view(LATENCY_VIEW)
    exporter = stats_exporter.new_stats_exporter()
    print('Exporting stats to project "{}"'.format(exporter.options.project_id))
    stats.stats.view_manager.register_exporter(exporter)
    for num in range(100):
        ms = random() * 5 * 1000
        mmap = stats.stats.stats_recorder.new_measurement_map()
        mmap.measure_float_put(LATENCY_MS, ms)
        mmap.record()
        print(f'Fake latency recorded ({num}: {ms})')
    time.sleep(65)
if __name__ == '__main__':
    main()