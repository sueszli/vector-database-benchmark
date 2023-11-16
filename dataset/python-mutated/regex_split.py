def regex_split(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        plants_split = pipeline | 'Garden plants' >> beam.Create(['🍓 : Strawberry : perennial', '🥕 : Carrot : biennial', '🍆\t:\tEggplant : perennial', '🍅 : Tomato : annual', '🥔 : Potato : perennial']) | 'Parse plants' >> beam.Regex.split('\\s*:\\s*') | beam.Map(print)
        if test:
            test(plants_split)
if __name__ == '__main__':
    regex_split()