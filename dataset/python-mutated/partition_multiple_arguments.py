def partition_multiple_arguments(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    import json

    def split_dataset(plant, num_partitions, ratio):
        if False:
            while True:
                i = 10
        assert num_partitions == len(ratio)
        bucket = sum(map(ord, json.dumps(plant))) % sum(ratio)
        total = 0
        for (i, part) in enumerate(ratio):
            total += part
            if bucket < total:
                return i
        return len(ratio) - 1
    with beam.Pipeline() as pipeline:
        (train_dataset, test_dataset) = pipeline | 'Gardening plants' >> beam.Create([{'icon': '🍓', 'name': 'Strawberry', 'duration': 'perennial'}, {'icon': '🥕', 'name': 'Carrot', 'duration': 'biennial'}, {'icon': '🍆', 'name': 'Eggplant', 'duration': 'perennial'}, {'icon': '🍅', 'name': 'Tomato', 'duration': 'annual'}, {'icon': '🥔', 'name': 'Potato', 'duration': 'perennial'}]) | 'Partition' >> beam.Partition(split_dataset, 2, ratio=[8, 2])
        train_dataset | 'Train' >> beam.Map(lambda x: print('train: {}'.format(x)))
        test_dataset | 'Test' >> beam.Map(lambda x: print('test: {}'.format(x)))
        if test:
            test(train_dataset, test_dataset)
if __name__ == '__main__':
    partition_multiple_arguments()