def regex_find_all(test=None):
    if False:
        return 10
    import apache_beam as beam
    regex = '(?P<icon>[^\\s,]+), *(\\w+), *(\\w+)'
    with beam.Pipeline() as pipeline:
        plants_find_all = pipeline | 'Garden plants' >> beam.Create(['# 🍓, Strawberry, perennial', '# 🥕, Carrot, biennial ignoring trailing words', '# 🍆, Eggplant, perennial - 🍌, Banana, perennial', '# 🍅, Tomato, annual - 🍉, Watermelon, annual', '# 🥔, Potato, perennial']) | 'Parse plants' >> beam.Regex.find_all(regex) | beam.Map(print)
        if test:
            test(plants_find_all)
if __name__ == '__main__':
    regex_find_all()