def regex_find_kv(test=None):
    if False:
        return 10
    import apache_beam as beam
    regex = '(?P<icon>[^\\s,]+), *(\\w+), *(\\w+)'
    with beam.Pipeline() as pipeline:
        plants_matches_kv = pipeline | 'Garden plants' >> beam.Create(['# 🍓, Strawberry, perennial', '# 🥕, Carrot, biennial ignoring trailing words', '# 🍆, Eggplant, perennial - 🍌, Banana, perennial', '# 🍅, Tomato, annual - 🍉, Watermelon, annual', '# 🥔, Potato, perennial']) | 'Parse plants' >> beam.Regex.find_kv(regex, keyGroup='icon') | beam.Map(print)
        if test:
            test(plants_matches_kv)
if __name__ == '__main__':
    regex_find_kv()