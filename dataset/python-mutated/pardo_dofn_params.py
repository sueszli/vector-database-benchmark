def pardo_dofn_params(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam

    class AnalyzeElement(beam.DoFn):

        def process(self, elem, timestamp=beam.DoFn.TimestampParam, window=beam.DoFn.WindowParam):
            if False:
                return 10
            yield '\n'.join(['# timestamp', 'type(timestamp) -> ' + repr(type(timestamp)), 'timestamp.micros -> ' + repr(timestamp.micros), 'timestamp.to_rfc3339() -> ' + repr(timestamp.to_rfc3339()), 'timestamp.to_utc_datetime() -> ' + repr(timestamp.to_utc_datetime()), '', '# window', 'type(window) -> ' + repr(type(window)), 'window.start -> {} ({})'.format(window.start, window.start.to_utc_datetime()), 'window.end -> {} ({})'.format(window.end, window.end.to_utc_datetime()), 'window.max_timestamp() -> {} ({})'.format(window.max_timestamp(), window.max_timestamp().to_utc_datetime())])
    with beam.Pipeline() as pipeline:
        dofn_params = pipeline | 'Create a single test element' >> beam.Create([':)']) | 'Add timestamp (Spring equinox 2020)' >> beam.Map(lambda elem: beam.window.TimestampedValue(elem, 1584675660)) | 'Fixed 30sec windows' >> beam.WindowInto(beam.window.FixedWindows(30)) | 'Analyze element' >> beam.ParDo(AnalyzeElement()) | beam.Map(print)
        if test:
            test(dofn_params)
if __name__ == '__main__':
    pardo_dofn_params()