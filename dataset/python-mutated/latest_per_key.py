def latest_per_key(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    import time

    def to_unix_time(time_str, format='%Y-%m-%d %H:%M:%S'):
        if False:
            return 10
        return time.mktime(time.strptime(time_str, format))
    with beam.Pipeline() as pipeline:
        latest_elements_per_key = pipeline | 'Create crops' >> beam.Create([('spring', {'item': '🥕', 'harvest': '2020-06-28 00:00:00'}), ('spring', {'item': '🍓', 'harvest': '2020-06-16 00:00:00'}), ('summer', {'item': '🥕', 'harvest': '2020-07-17 00:00:00'}), ('summer', {'item': '🍓', 'harvest': '2020-08-26 00:00:00'}), ('summer', {'item': '🍆', 'harvest': '2020-09-04 00:00:00'}), ('summer', {'item': '🥬', 'harvest': '2020-09-18 00:00:00'}), ('summer', {'item': '🍅', 'harvest': '2020-09-22 00:00:00'}), ('autumn', {'item': '🍅', 'harvest': '2020-10-01 00:00:00'}), ('autumn', {'item': '🥬', 'harvest': '2020-10-20 00:00:00'}), ('autumn', {'item': '🍆', 'harvest': '2020-10-26 00:00:00'}), ('winter', {'item': '🥬', 'harvest': '2020-02-24 00:00:00'})]) | 'With timestamps' >> beam.Map(lambda pair: beam.window.TimestampedValue((pair[0], pair[1]['item']), to_unix_time(pair[1]['harvest']))) | 'Get latest elements per key' >> beam.combiners.Latest.PerKey() | beam.Map(print)
        if test:
            test(latest_elements_per_key)
if __name__ == '__main__':
    latest_per_key()