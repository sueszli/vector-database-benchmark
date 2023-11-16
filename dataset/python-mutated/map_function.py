def map_function(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam

    def strip_header_and_newline(text):
        if False:
            while True:
                i = 10
        return text.strip('# \n')
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Gardening plants' >> beam.Create(['# 🍓Strawberry\n', '# 🥕Carrot\n', '# 🍆Eggplant\n', '# 🍅Tomato\n', '# 🥔Potato\n']) | 'Strip header' >> beam.Map(strip_header_and_newline) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    map_function()