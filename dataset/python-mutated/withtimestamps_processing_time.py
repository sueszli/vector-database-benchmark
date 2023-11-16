def withtimestamps_processing_time(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    import time

    class GetTimestamp(beam.DoFn):

        def process(self, plant, timestamp=beam.DoFn.TimestampParam):
            if False:
                i = 10
                return i + 15
            yield '{} - {}'.format(timestamp.to_utc_datetime(), plant['name'])
    with beam.Pipeline() as pipeline:
        plant_processing_times = pipeline | 'Garden plants' >> beam.Create([{'name': 'Strawberry'}, {'name': 'Carrot'}, {'name': 'Artichoke'}, {'name': 'Tomato'}, {'name': 'Potato'}]) | 'With timestamps' >> beam.Map(lambda plant: beam.window.TimestampedValue(plant, time.time())) | 'Get timestamp' >> beam.ParDo(GetTimestamp()) | beam.Map(print)
        if test:
            test(plant_processing_times)
if __name__ == '__main__':
    withtimestamps_processing_time()