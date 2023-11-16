def regex_replace_all(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        plants_replace_all = pipeline | 'Garden plants' >> beam.Create(['🍓 : Strawberry : perennial', '🥕 : Carrot : biennial', '🍆\t:\tEggplant\t:\tperennial', '🍅 : Tomato : annual', '🥔 : Potato : perennial']) | 'To CSV' >> beam.Regex.replace_all('\\s*:\\s*', ',') | beam.Map(print)
        if test:
            test(plants_replace_all)
if __name__ == '__main__':
    regex_replace_all()