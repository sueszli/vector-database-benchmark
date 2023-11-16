def regex_find(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    regex = '(?P<icon>[^\\s,]+), *(\\w+), *(\\w+)'
    with beam.Pipeline() as pipeline:
        plants_matches = pipeline | 'Garden plants' >> beam.Create(['# 🍓, Strawberry, perennial', '# 🥕, Carrot, biennial ignoring trailing words', '# 🍆, Eggplant, perennial - 🍌, Banana, perennial', '# 🍅, Tomato, annual - 🍉, Watermelon, annual', '# 🥔, Potato, perennial']) | 'Parse plants' >> beam.Regex.find(regex) | beam.Map(print)
        if test:
            test(plants_matches)
if __name__ == '__main__':
    regex_find()